[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "Preflight",
        "Smoke",
        "PrepareValidation",
        "RunValidation",
        "AuditValidation",
        "PrepareFrozenTest",
        "FrozenTest"
    )]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,

    [string]$ExpectedSourceState = "",

    [string]$SelectedVariant = "",

    [int]$Gpu = 0,

    [string]$PythonCommand = "python",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AllowedVariants = @(
    "E0_final_only",
    "E1_multistep",
    "E2_multistep_random_horizon",
    "E3_sgd_residual"
)
$StartedAt = (Get-Date).ToUniversalTime()
$StageClock = [System.Diagnostics.Stopwatch]::StartNew()
$InvocationLine = $MyInvocation.Line
$Gate = $null
$Plan = $null

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$PyArgs)

    & $PythonCommand @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $PythonCommand $($PyArgs -join ' ')"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-RepositoryRoot {
    $Required = @(
        "train_meta.py",
        "configs/base.yaml",
        "configs/datasets/cicids2017.yaml",
        "scripts/generate_eval_task_manifest.py",
        "scripts/run_experiments.py",
        "src/utils/experiment_protocol.py"
    )
    foreach ($Path in $Required) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Run from the repository root. Missing tracked file: $Path"
        }
    }
}

function Assert-CleanExecutionGate {
    $GateArgs = @(
        "-c",
        @"
import json
from src.utils.provenance import assert_execution_gate
state = assert_execution_gate(
    "$ExpectedCommit",
    expected_source_state_sha256=("$ExpectedSourceState" or None),
)
print(json.dumps(state, ensure_ascii=False))
"@
    )
    $Raw = & $PythonCommand @GateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Git/source-state execution gate failed."
    }
    return ($Raw | Out-String | ConvertFrom-Json)
}

function Get-ProtocolPlan {
    $Args = @(
        "-m", "src.utils.experiment_protocol",
        "--stage", $Stage,
        "--root", $Root,
        "--expected-commit", $ExpectedCommit
    )
    if ($SelectedVariant) {
        $Args += @("--selected-variant", $SelectedVariant)
    }
    $Raw = & $PythonCommand @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to expand the protocol plan."
    }
    return ($Raw | Out-String | ConvertFrom-Json)
}

function Assert-OutputNamespace {
    $Short = $ExpectedCommit.Substring(0, [Math]::Min(7, $ExpectedCommit.Length))
    $ExpectedToken = "metaopt_remediation_clean_$Short"
    $ResolvedParent = Resolve-Path -LiteralPath (Split-Path -Parent $Root) -ErrorAction SilentlyContinue
    $Leaf = Split-Path -Leaf $Root
    if ($Leaf -ne $ExpectedToken) {
        throw "Output root must be a new clean-rerun namespace named $ExpectedToken"
    }
    if ($null -eq $ResolvedParent) {
        throw "Output root parent does not exist: $(Split-Path -Parent $Root)"
    }
}

function Assert-CudaAvailable {
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    Invoke-Python -PyArgs @(
        "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
    )
}

function Assert-NotCompleted {
    param([Parameter(Mandatory = $true)][string]$ReceiptPath)

    if (Test-Path -LiteralPath $ReceiptPath -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
        if ($Existing.completion_status -eq "success") {
            throw "Stage already completed; refusing repeat execution: $ReceiptPath"
        }
        throw "A prior non-success receipt exists; audit it before retrying: $ReceiptPath"
    }
}

function Assert-NewRunDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite an existing run path: $Path"
    }
}

function Write-StageReceipt {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("success", "failed", "interrupted")]
        [string]$Status,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [string[]]$RequiredArtifacts = @(),
        [string]$ErrorMessage = ""
    )

    $FinishedAt = (Get-Date).ToUniversalTime()
    $Missing = @(
        $RequiredArtifacts |
            Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    )
    if ($Status -eq "success" -and ($ExitCode -ne 0 -or $Missing.Count -gt 0)) {
        throw "Cannot write success receipt; missing artifacts: $($Missing -join ', ')"
    }
    $TrackedHashes = @{}
    if ($null -ne $Gate) {
        foreach ($Property in $Gate.tracked_source_file_sha256.PSObject.Properties) {
            $TrackedHashes[$Property.Name] = $Property.Value
        }
    }
    $RuntimeRaw = & $PythonCommand -c @"
import json
from src.utils.provenance import runtime_environment
print(json.dumps(runtime_environment(), ensure_ascii=False))
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to record the Python runtime environment."
    }
    $Runtime = $RuntimeRaw | Out-String | ConvertFrom-Json
    $Artifacts = @(
        foreach ($Path in $RequiredArtifacts) {
            [ordered]@{
                path = [System.IO.Path]::GetFullPath($Path)
                exists = Test-Path -LiteralPath $Path -PathType Leaf
                sha256 = Get-Sha256 $Path
            }
        }
    )
    $ManifestRows = @(
        if ($null -ne $Plan) {
            $Plan.jobs |
                Where-Object { $_.manifest_relative } |
                ForEach-Object {
                    $Path = Join-Path $Root $_.manifest_relative
                    [ordered]@{
                        path = [System.IO.Path]::GetFullPath($Path)
                        sha256 = Get-Sha256 $Path
                        sidecar_sha256 = Get-Sha256 "$Path.sha256"
                    }
                } |
                Sort-Object path -Unique
        }
    )
    $RunStateRows = @(
        if ($null -ne $Plan) {
            $Plan.jobs |
                Where-Object { $_.variant -and $_.output_relative } |
                ForEach-Object {
                    $RunDir = Join-Path $Root $_.output_relative
                    [ordered]@{
                        variant = $_.variant
                        attack = $_.attack
                        seed = $_.seed
                        effective_config_sha256 = Get-Sha256 (
                            Join-Path $RunDir "effective_config.json"
                        )
                        artifact_sha256 = Get-Sha256 (
                            Join-Path $RunDir "meta_artifacts.pt"
                        )
                        checkpoint_sha256 = Get-Sha256 (
                            Join-Path $RunDir "checkpoints/best.pt"
                        )
                    }
                }
        }
    )
    $RunProtocolReceipts = @(
        if ($null -ne $Plan) {
            $Plan.jobs |
                Where-Object { $_.output_relative } |
                ForEach-Object {
                    $Path = Join-Path $Root (
                        "$($_.output_relative)/protocol_run_completion_receipt.json"
                    )
                    if (Test-Path -LiteralPath $Path -PathType Leaf) {
                        Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
                    }
                }
        }
    )
    $Payload = [ordered]@{
        receipt_schema_version = 1
        stage = $Stage
        completion_status = $Status
        success = $Status -eq "success"
        full_git_commit = if ($null -ne $Gate) { $Gate.git_commit } else { $null }
        parent_commit = if ($null -ne $Gate) { $Gate.parent_commit } else { $null }
        worktree_clean = if ($null -ne $Gate) { $Gate.worktree_clean } else { $null }
        source_state_sha256 = if ($null -ne $Gate) { $Gate.source_state_sha256 } else { $null }
        source_state_algorithm = if ($null -ne $Gate) { $Gate.source_state_algorithm } else { $null }
        source_state_commit = if ($null -ne $Gate) { $Gate.source_state_commit } else { $null }
        tracked_source_file_sha256 = $TrackedHashes
        tracked_source_git_blob_oid = if ($null -ne $Gate) {
            $Gate.tracked_source_git_blob_oid
        } else { @{} }
        worktree_source_state_sha256 = if ($null -ne $Gate) {
            $Gate.worktree_source_state_sha256
        } else { $null }
        worktree_source_state_algorithm = if ($null -ne $Gate) {
            $Gate.worktree_source_state_algorithm
        } else { $null }
        worktree_source_file_sha256 = if ($null -ne $Gate) {
            $Gate.worktree_source_file_sha256
        } else { @{} }
        plan = $Plan
        effective_config_hashes = $RunStateRows
        dataset_fingerprints = @(
            $RunProtocolReceipts |
                ForEach-Object { $_.dataset_fingerprint } |
                Where-Object { $_ } |
                Sort-Object -Unique
        )
        manifest_hashes = $ManifestRows
        theta0_hashes = @(
            $RunProtocolReceipts |
                ForEach-Object { $_.theta0_sha256 } |
                Where-Object { $_ } |
                Sort-Object -Unique
        )
        checkpoint_hashes = $RunStateRows
        dataset_role = switch ($Stage) {
            "Smoke" { "validation" }
            "PrepareValidation" { "validation" }
            "RunValidation" { "validation" }
            "PrepareFrozenTest" { "test" }
            "FrozenTest" { "test" }
            default { "none" }
        }
        validation_dataset_constructed = $Stage -in @("Smoke", "PrepareValidation", "RunValidation")
        validation_dataset_accessed = $Stage -in @("Smoke", "PrepareValidation", "RunValidation")
        test_dataset_constructed = $Stage -in @("PrepareFrozenTest", "FrozenTest")
        test_dataset_accessed = $Stage -in @("PrepareFrozenTest", "FrozenTest")
        metaopt_training_ran = $Stage -in @("Smoke", "RunValidation")
        command_line = $InvocationLine
        hostname = [System.Net.Dns]::GetHostName()
        environment = $Runtime
        started_at_utc = $StartedAt.ToString("o")
        finished_at_utc = $FinishedAt.ToString("o")
        duration_seconds = $StageClock.Elapsed.TotalSeconds
        exit_code = $ExitCode
        required_artifacts = $Artifacts
        output_paths = @([System.IO.Path]::GetFullPath($Root))
        error = if ($ErrorMessage) { $ErrorMessage } else { $null }
    }
    $ReceiptDir = Join-Path $Root "_stage_receipts"
    New-Item -ItemType Directory -Force -Path $ReceiptDir | Out-Null
    $ReceiptPath = Join-Path $ReceiptDir "$Stage.json"
    $Payload | ConvertTo-Json -Depth 20 |
        Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
}

function Write-RunReceipt {
    param(
        [Parameter(Mandatory = $true)]$Job,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)]
        [ValidateSet("success", "failed", "interrupted")]
        [string]$Status,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][datetime]$RunStartedAt,
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Stopwatch]$RunClock,
        [string]$ErrorMessage = ""
    )

    $Artifact = Join-Path $RunDir "meta_artifacts.pt"
    $EffectiveConfig = Join-Path $RunDir "effective_config.json"
    $Best = Join-Path $RunDir "checkpoints/best.pt"
    $Manifest = Join-Path $Root $Job.manifest_relative
    $Sidecar = "$Manifest.sha256"
    $EvaluationDir = Join-Path $RunDir "validation"
    $EvaluationReceiptPath = Join-Path $EvaluationDir "completion_receipt.json"
    $TrainingReceiptPath = Join-Path $RunDir "training_completion_receipt.json"
    $Required = @(
        $Artifact,
        $EffectiveConfig,
        $Best,
        $Manifest,
        $Sidecar,
        $TrainingReceiptPath,
        $EvaluationReceiptPath,
        (Join-Path $EvaluationDir "results.json"),
        (Join-Path $EvaluationDir "prediction_trajectories.npz"),
        (Join-Path $EvaluationDir "update_analysis.csv")
    )
    $Missing = @($Required | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    })
    if ($Status -eq "success" -and ($ExitCode -ne 0 -or $Missing.Count -gt 0)) {
        throw "Cannot mark run successful; missing: $($Missing -join ', ')"
    }
    $EvaluationReceipt = $null
    if (Test-Path -LiteralPath $EvaluationReceiptPath -PathType Leaf) {
        $EvaluationReceipt = (
            Get-Content -LiteralPath $EvaluationReceiptPath -Raw |
                ConvertFrom-Json
        )
    }
    $Payload = [ordered]@{
        receipt_schema_version = 1
        stage = if ($Job.kind -eq "smoke_validation_run") {
            "smoke_validation_run"
        } else {
            "formal_validation_run"
        }
        completion_status = $Status
        success = $Status -eq "success"
        variant = $Job.variant
        attack = $Job.attack
        seed = $Job.seed
        shot = $Job.shot
        horizon = $Job.horizon
        full_git_commit = $Gate.git_commit
        parent_commit = $Gate.parent_commit
        worktree_clean = $Gate.worktree_clean
        source_state_sha256 = $Gate.source_state_sha256
        source_state_algorithm = $Gate.source_state_algorithm
        source_state_commit = $Gate.source_state_commit
        tracked_source_file_sha256 = $Gate.tracked_source_file_sha256
        tracked_source_git_blob_oid = $Gate.tracked_source_git_blob_oid
        worktree_source_state_sha256 = $Gate.worktree_source_state_sha256
        worktree_source_state_algorithm = $Gate.worktree_source_state_algorithm
        worktree_source_file_sha256 = $Gate.worktree_source_file_sha256
        effective_config_path = [System.IO.Path]::GetFullPath($EffectiveConfig)
        effective_config_sha256 = Get-Sha256 $EffectiveConfig
        dataset_role = "validation"
        dataset_fingerprint = if ($null -ne $EvaluationReceipt) {
            $EvaluationReceipt.dataset_fingerprint
        } else { $null }
        manifest_path = [System.IO.Path]::GetFullPath($Manifest)
        manifest_sha256 = Get-Sha256 $Manifest
        sidecar_path = [System.IO.Path]::GetFullPath($Sidecar)
        sidecar_sha256 = Get-Sha256 $Sidecar
        theta0_sha256 = if ($null -ne $EvaluationReceipt) {
            $EvaluationReceipt.theta0_sha256
        } else { $null }
        artifact_path = [System.IO.Path]::GetFullPath($Artifact)
        artifact_sha256 = Get-Sha256 $Artifact
        checkpoint_path = [System.IO.Path]::GetFullPath($Best)
        checkpoint_sha256 = Get-Sha256 $Best
        environment = if ($null -ne $EvaluationReceipt) {
            $EvaluationReceipt.environment
        } else { $null }
        hostname = [System.Net.Dns]::GetHostName()
        command_line = $InvocationLine
        started_at_utc = $RunStartedAt.ToUniversalTime().ToString("o")
        finished_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        duration_seconds = $RunClock.Elapsed.TotalSeconds
        exit_code = $ExitCode
        validation_dataset_constructed = $true
        validation_dataset_accessed = $true
        test_dataset_constructed = $false
        test_dataset_accessed = $false
        metaopt_training_ran = $true
        output_paths = @([System.IO.Path]::GetFullPath($RunDir))
        required_artifacts = @(
            foreach ($Path in $Required) {
                [ordered]@{
                    path = [System.IO.Path]::GetFullPath($Path)
                    exists = Test-Path -LiteralPath $Path -PathType Leaf
                    sha256 = Get-Sha256 $Path
                }
            }
        )
        error = if ($ErrorMessage) { $ErrorMessage } else { $null }
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $Payload | ConvertTo-Json -Depth 20 |
        Set-Content -LiteralPath (
            Join-Path $RunDir "protocol_run_completion_receipt.json"
        ) -Encoding UTF8
}

function Invoke-ManifestGeneration {
    param(
        [Parameter(Mandatory = $true)]$Job,
        [Parameter(Mandatory = $true)][ValidateSet("validation", "test")]
        [string]$Split,
        [Parameter(Mandatory = $true)][int]$Tasks
    )

    $Manifest = Join-Path $Root $Job.manifest_relative
    $Attack = [string]$Job.attack
    $Seed = [int]$Job.seed
    $TaskSeed = if ($Split -eq "validation") { $Seed + 1 } else { $Seed + 1001 }
    $BaseArtifact = if ($Job.kind -eq "smoke_validation_run") {
        Join-Path $Root "$($Job.output_relative)/meta_artifacts.pt"
    } else {
        Join-Path $Root (
            "validation/E0_final_only/$Attack/seed_$Seed/horizon_20/meta_artifacts.pt"
        )
    }
    $Args = @(
        "scripts/generate_eval_task_manifest.py",
        "--config", "configs/base.yaml",
        "--dataset", "configs/datasets/cicids2017.yaml",
        "--base-checkpoint-path", $BaseArtifact,
        "--out", $Manifest,
        "--shot", "5",
        "--tasks", "$Tasks",
        "--task-seed", "$TaskSeed",
        "--split", $Split,
        "--expected-commit", $ExpectedCommit
    )
    if ($ExpectedSourceState) {
        $Args += @("--expected-source-state", $ExpectedSourceState)
    }
    $Args += "--override"
    $Args += @($Job.overrides)
    Invoke-Python -PyArgs $Args
}

function Invoke-ValidationRun {
    param(
        [Parameter(Mandatory = $true)]$Job,
        [Parameter(Mandatory = $true)][bool]$Smoke
    )

    $RunStartedAt = (Get-Date).ToUniversalTime()
    $RunClock = [System.Diagnostics.Stopwatch]::StartNew()
    $RunDir = Join-Path $Root $Job.output_relative
    try {
        Assert-NewRunDirectory $RunDir
        $Artifact = Join-Path $RunDir "meta_artifacts.pt"
        $Manifest = Join-Path $Root $Job.manifest_relative
        if ($Smoke) {
            Invoke-ManifestGeneration -Job $Job -Split "validation" -Tasks 2
        } elseif (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
            throw "Prepared validation manifest is missing: $Manifest"
        }
        $TrainArgs = @(
        "train_meta.py",
        "--config", "configs/base.yaml",
        "--dataset", "configs/datasets/cicids2017.yaml",
        "--out", $Artifact,
        "--expected-commit", $ExpectedCommit
    )
        if ($ExpectedSourceState) {
            $TrainArgs += @("--expected-source-state", $ExpectedSourceState)
        }
        $TrainArgs += "--override"
        $TrainArgs += "device.prefer=cuda:0"
        $TrainArgs += @($Job.overrides)
        Invoke-Python -PyArgs $TrainArgs
        Invoke-Python -PyArgs @(
        "scripts/verify_metaopt_checkpoint.py",
        "--artifacts", $Artifact,
        "--best", (Join-Path $RunDir "checkpoints/best.pt"),
        "--last", (Join-Path $RunDir "checkpoints/last.pt"),
        "--out", (Join-Path $RunDir "p0_checkpoint_scope_receipt.json")
        )
        $EvaluationDir = Join-Path $RunDir "validation"
        $EvalArgs = @(
        "scripts/run_experiments.py",
        "--artifacts", $Artifact,
        "--out", $EvaluationDir,
        "--phase", "validation",
        "--validation-task-manifest", $Manifest,
        "--expected-commit", $ExpectedCommit
        )
        if ($ExpectedSourceState) {
            $EvalArgs += @("--expected-source-state", $ExpectedSourceState)
        }
        $EvalArgs += @(
        "--override",
        "device.prefer=cuda:0",
        "compare.shots=[5]",
        "compare.val_tasks=30",
        "adaptation_speed.max_steps=20",
        "adaptation_speed.checkpoints=[0,1,2,5,10,20]",
        "compare.baseline_lr_grid.sgd=[0.5,0.1,0.05,0.01]",
        "compare.baseline_lr_grid.adam=[0.001,0.003,0.01,0.03,0.1,0.3]"
        )
        Invoke-Python -PyArgs $EvalArgs
        $Completion = Join-Path $EvaluationDir "completion_receipt.json"
        $Receipt = Get-Content -LiteralPath $Completion -Raw | ConvertFrom-Json
        if (
            -not $Receipt.success -or
            $Receipt.dataset_role -ne "validation" -or
            $Receipt.test_dataset_constructed -or
            $Receipt.test_dataset_accessed
        ) {
            throw "Validation completion receipt violates split isolation: $Completion"
        }
        Write-RunReceipt `
            -Job $Job `
            -RunDir $RunDir `
            -Status "success" `
            -ExitCode 0 `
            -RunStartedAt $RunStartedAt `
            -RunClock $RunClock
    } catch [System.Management.Automation.PipelineStoppedException] {
        Write-RunReceipt `
            -Job $Job `
            -RunDir $RunDir `
            -Status "interrupted" `
            -ExitCode 130 `
            -RunStartedAt $RunStartedAt `
            -RunClock $RunClock `
            -ErrorMessage $_.Exception.Message
        throw
    } catch {
        Write-RunReceipt `
            -Job $Job `
            -RunDir $RunDir `
            -Status "failed" `
            -ExitCode 1 `
            -RunStartedAt $RunStartedAt `
            -RunClock $RunClock `
            -ErrorMessage $_.Exception.Message
        throw
    }
}

Assert-RepositoryRoot
$Plan = Get-ProtocolPlan

if ($DryRun) {
    $Plan | ConvertTo-Json -Depth 20
    Write-Host "DRY_RUN=PASS; no gate, dataset, model, CUDA, output, validation, or test was accessed."
    exit 0
}

Assert-OutputNamespace
$Gate = Assert-CleanExecutionGate
if (-not (Test-Path -LiteralPath $Root)) {
    New-Item -ItemType Directory -Path $Root | Out-Null
}
$StageReceipt = Join-Path $Root "_stage_receipts/$Stage.json"
Assert-NotCompleted $StageReceipt

try {
    switch ($Stage) {
        "Preflight" {
            if ($Plan.job_count -ne 0) {
                throw "Preflight must not expand experiment jobs."
            }
            Write-StageReceipt -Status "success" -ExitCode 0
        }
        "Smoke" {
            if ($Plan.job_count -ne 1) {
                throw "Smoke must expand exactly one run."
            }
            Assert-CudaAvailable
            Invoke-ValidationRun -Job $Plan.jobs[0] -Smoke $true
            $Required = @(
                (Join-Path $Root "$($Plan.jobs[0].output_relative)/meta_artifacts.pt"),
                (Join-Path $Root "$($Plan.jobs[0].output_relative)/validation/results.json"),
                (Join-Path $Root "$($Plan.jobs[0].output_relative)/validation/prediction_trajectories.npz"),
                (Join-Path $Root "$($Plan.jobs[0].output_relative)/validation/update_analysis.csv"),
                (Join-Path $Root "$($Plan.jobs[0].output_relative)/protocol_run_completion_receipt.json")
            )
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
        "PrepareValidation" {
            if ($Plan.job_count -ne 4) {
                throw "PrepareValidation must expand four manifests and no training."
            }
            foreach ($Job in $Plan.jobs) {
                Invoke-ManifestGeneration -Job $Job -Split "validation" -Tasks 30
            }
            $Required = @($Plan.jobs | ForEach-Object { Join-Path $Root $_.manifest_relative })
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
        "RunValidation" {
            if ($Plan.job_count -ne 16) {
                throw "RunValidation must expand exactly 16 pre-registered runs."
            }
            Assert-CudaAvailable
            foreach ($Job in $Plan.jobs) {
                Invoke-ValidationRun -Job $Job -Smoke $false
            }
            $Required = @(
                $Plan.jobs | ForEach-Object {
                    Join-Path $Root "$($_.output_relative)/validation/completion_receipt.json"
                    Join-Path $Root "$($_.output_relative)/protocol_run_completion_receipt.json"
                }
            )
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
        "AuditValidation" {
            Invoke-Python -PyArgs @(
                "scripts/audit_clean_validation.py",
                "--root", $Root,
                "--expected-commit", $ExpectedCommit
            )
            $Required = @(
                (Join-Path $Root "validation_summary.csv"),
                (Join-Path $Root "validation_scorecard.csv"),
                (Join-Path $Root "botnet_task_independence.csv"),
                (Join-Path $Root "validation_gate_receipt.json")
            )
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
        "PrepareFrozenTest" {
            if ($SelectedVariant -notin $AllowedVariants) {
                throw "PrepareFrozenTest requires a pre-registered -SelectedVariant."
            }
            $GateReceipt = Join-Path $Root "validation_gate_receipt.json"
            if (-not (Test-Path -LiteralPath $GateReceipt -PathType Leaf)) {
                throw "Validation gate receipt is missing."
            }
            $ValidationGate = Get-Content -LiteralPath $GateReceipt -Raw | ConvertFrom-Json
            if (
                -not $ValidationGate.validation_gate_passed -or
                $ValidationGate.selected_variant -ne $SelectedVariant
            ) {
                throw "Selected variant does not match the passed validation gate."
            }
            foreach ($Job in $Plan.jobs) {
                Invoke-ManifestGeneration -Job $Job -Split "test" -Tasks 100
            }
            Invoke-Python -PyArgs @(
                "scripts/prepare_frozen_test_receipt.py",
                "--root", $Root,
                "--expected-commit", $ExpectedCommit,
                "--selected-variant", $SelectedVariant
            )
            $Required = @(
                $Plan.jobs | ForEach-Object { Join-Path $Root $_.manifest_relative }
            ) + @(Join-Path $Root "frozen_test/frozen_experiment_receipt.json")
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
        "FrozenTest" {
            if ($env:ALLOW_FROZEN_TEST -ne "YES") {
                throw "FrozenTest requires explicit ALLOW_FROZEN_TEST=YES approval."
            }
            $FrozenReceipt = Join-Path $Root "frozen_test/frozen_experiment_receipt.json"
            if (-not (Test-Path -LiteralPath $FrozenReceipt -PathType Leaf)) {
                throw "Frozen experiment receipt is missing."
            }
            $Frozen = Get-Content -LiteralPath $FrozenReceipt -Raw | ConvertFrom-Json
            if (
                $Frozen.selected_variant -ne $SelectedVariant -or
                -not $Frozen.validation_gate_passed
            ) {
                throw "Frozen receipt selection/gate mismatch."
            }
            Assert-CudaAvailable
            foreach ($Job in $Plan.jobs) {
                $RunDir = Join-Path $Root $Job.output_relative
                Assert-NewRunDirectory $RunDir
                $ValidationDir = Join-Path $Root (
                    "validation/$($Job.variant)/$($Job.attack)/seed_$($Job.seed)/horizon_20"
                )
                $Args = @(
                    "scripts/run_experiments.py",
                    "--artifacts", (Join-Path $ValidationDir "meta_artifacts.pt"),
                    "--out", $RunDir,
                    "--phase", "test",
                    "--selection-receipt", (Join-Path $Root (
                        "frozen_test/selections/$($Job.variant)/$($Job.attack)/" +
                        "seed_$($Job.seed)/validation_selection_frozen.json"
                    )),
                    "--test-task-manifest", (Join-Path $Root $Job.manifest_relative),
                    "--expected-commit", $ExpectedCommit,
                    "--override",
                    "device.prefer=cuda:0",
                    "compare.shots=[5]",
                    "adaptation_speed.max_steps=20",
                    "adaptation_speed.checkpoints=[0,1,2,5,10,20]"
                )
                if ($ExpectedSourceState) {
                    $Insert = [Array]::IndexOf($Args, "--override")
                    $Args = @($Args[0..($Insert - 1)]) +
                        @("--expected-source-state", $ExpectedSourceState) +
                        @($Args[$Insert..($Args.Count - 1)])
                }
                Invoke-Python -PyArgs $Args
            }
            $Required = @(
                $Plan.jobs | ForEach-Object {
                    Join-Path $Root "$($_.output_relative)/completion_receipt.json"
                }
            )
            Write-StageReceipt -Status "success" -ExitCode 0 -RequiredArtifacts $Required
        }
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    Write-StageReceipt -Status "interrupted" -ExitCode 130 -ErrorMessage $_.Exception.Message
    throw
} catch {
    Write-StageReceipt -Status "failed" -ExitCode 1 -ErrorMessage $_.Exception.Message
    throw
}
