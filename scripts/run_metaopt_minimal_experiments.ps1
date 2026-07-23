[CmdletBinding()]
param(
    [ValidateSet("Check", "Prepare", "Test")]
    [string]$Stage = "Check",

    [string]$Root = "outputs/metaopt_remediation",

    [string]$ChosenVariant = "",

    [int]$Gpu = 0,

    [string]$PythonCommand = "python",

    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AllowedVariants = @(
    "E0_final_only",
    "E1_multistep",
    "E2_multistep_random_horizon",
    "E3_sgd_residual"
)

$KnownClasses = @{
    ddos   = "['botnet','bruteforce','dos','heartbleed','infiltration','portscan','webattack']"
    botnet = "['bruteforce','ddos','dos','heartbleed','infiltration','portscan','webattack']"
}

$Variants = @(
    @{
        Id = "E0_final_only"
        Extra = @(
            "meta.query_objective.mode=final_only",
            "meta.random_horizon.enabled=false",
            "meta.mixed_shot.enabled=false",
            "meta_optimizer.update_mode=learned_delta"
        )
    },
    @{
        Id = "E1_multistep"
        Extra = @(
            "meta.query_objective.mode=multi_step",
            "meta.query_objective.supervised_steps=[1,2,5,10,20]",
            "meta.query_objective.weighting=early_heavy",
            "meta.query_objective.early_heavy_power=0.5",
            "meta.random_horizon.enabled=false",
            "meta.mixed_shot.enabled=false",
            "meta_optimizer.update_mode=learned_delta",
            "train.validation.selection_metric=curve_auc"
        )
    },
    @{
        Id = "E2_multistep_random_horizon"
        Extra = @(
            "meta.query_objective.mode=multi_step",
            "meta.query_objective.supervised_steps=[1,2,5,10,20]",
            "meta.query_objective.weighting=early_heavy",
            "meta.query_objective.early_heavy_power=0.5",
            "meta.random_horizon.enabled=true",
            "meta.random_horizon.min_steps=1",
            "meta.random_horizon.max_steps=20",
            "meta.mixed_shot.enabled=false",
            "meta_optimizer.update_mode=learned_delta",
            "train.validation.selection_metric=curve_auc"
        )
    },
    @{
        Id = "E3_sgd_residual"
        Extra = @(
            "meta.query_objective.mode=multi_step",
            "meta.query_objective.supervised_steps=[1,2,5,10,20]",
            "meta.query_objective.weighting=early_heavy",
            "meta.query_objective.early_heavy_power=0.5",
            "meta.random_horizon.enabled=false",
            "meta.mixed_shot.enabled=false",
            "meta_optimizer.update_mode=sgd_residual",
            "meta_optimizer.anchor_lr=0.1",
            "meta_optimizer.learnable_anchor_lr=false",
            "meta_optimizer.residual_enabled=true",
            "meta_optimizer.residual_zero_init=true",
            "meta_optimizer.gate_init=0.01",
            "meta_optimizer.learnable_gate=true",
            "meta_optimizer.trust_region_factor=2.0",
            "train.validation.selection_metric=curve_auc"
        )
    }
)

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$PyArgs)

    & $PythonCommand @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $PythonCommand $($PyArgs -join ' ')"
    }
}

function Assert-RepositoryRoot {
    $Required = @(
        "train_meta.py",
        "configs/base.yaml",
        "configs/datasets/cicids2017.yaml",
        "scripts/run_experiments.py",
        "scripts/generate_eval_task_manifest.py"
    )
    foreach ($Path in $Required) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Run this script from the repository root. Missing: $Path"
        }
    }
}

function Assert-Cuda {
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"

    $PythonPath = & $PythonCommand -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot execute Python command: $PythonCommand"
    }
    $TorchPresent = & $PythonCommand -c "import importlib.util; print('yes' if importlib.util.find_spec('torch') else 'no')"
    if ($LASTEXITCODE -ne 0 -or "$TorchPresent".Trim() -ne "yes") {
        throw @"
The selected Python interpreter does not contain PyTorch:
  $PythonPath

Activate the project's Conda/virtual environment, or install the CUDA build of
PyTorch into this exact interpreter. Then rerun with:
  -PythonCommand "$PythonPath"
"@
    }

    Invoke-Python -PyArgs @(
        "-c",
        "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda)"
    )
    $CudaPresent = & $PythonCommand -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
    if ($LASTEXITCODE -ne 0 -or "$CudaPresent".Trim() -ne "yes") {
        throw @"
PyTorch is installed in $PythonPath, but this build cannot access CUDA.
Install a CUDA-enabled PyTorch build compatible with the server driver, then
rerun Stage Check. Do not start the experiment with a CPU-only torch build.
"@
    }
    Invoke-Python -PyArgs @(
        "-c",
        "import torch; print('CUDA ready:', torch.cuda.get_device_name(0))"
    )
}

function Assert-CanCreateOrResume {
    param([string]$Path, [string]$Description)

    if (Test-Path -LiteralPath $Path) {
        if ($Resume) {
            Write-Host "RESUME: $Description already exists: $Path" -ForegroundColor DarkYellow
            return $false
        }
        throw "$Description already exists: $Path. Use -Resume only to continue the same run; no files will be deleted."
    }
    return $true
}

function Get-Median {
    param([double[]]$Values)

    $Sorted = @($Values | Sort-Object)
    if ($Sorted.Count -eq 0) {
        return [double]::NaN
    }
    $Middle = [int][Math]::Floor($Sorted.Count / 2)
    if (($Sorted.Count % 2) -eq 1) {
        return [double]$Sorted[$Middle]
    }
    return ([double]$Sorted[$Middle - 1] + [double]$Sorted[$Middle]) / 2.0
}

function ConvertTo-StrictBoolean {
    param($Value)

    if ($Value -is [bool]) {
        return $Value
    }
    return [System.Boolean]::Parse("$Value")
}

function Write-ValidationReports {
    $Rows = foreach ($Variant in $AllowedVariants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $ResultPath = "$Root/$Variant/$Attack/seed_$Seed/horizon_20/validation/results.json"
                $Json = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
                $Exp = $Json.exp1_5shot
                $Meta = $Exp.methods.MetaOpt
                $SGD = $Exp.methods.SGD
                $Adam = $Exp.baseline_lr_validation.Adam

                [PSCustomObject]@{
                    Variant = $Variant
                    Attack = $Attack
                    Seed = $Seed
                    MetaStep1MacroF1 = [double]$Meta.adaptation_analysis.checkpoints.'1'.macro_f1.mean
                    SGDStep1MacroF1 = [double]$SGD.adaptation_analysis.checkpoints.'1'.macro_f1.mean
                    MetaMinusSGDStep1 = (
                        [double]$Meta.adaptation_analysis.checkpoints.'1'.macro_f1.mean -
                        [double]$SGD.adaptation_analysis.checkpoints.'1'.macro_f1.mean
                    )
                    MetaCurveAUC = [double]$Meta.adaptation_analysis.curve_auc_mean
                    SGDCurveAUC = [double]$SGD.adaptation_analysis.curve_auc_mean
                    MetaMinusSGDCurveAUC = (
                        [double]$Meta.adaptation_analysis.curve_auc_mean -
                        [double]$SGD.adaptation_analysis.curve_auc_mean
                    )
                    MetaFinalMacroF1 = [double]$Meta.final_metrics_avg_per_task.macro_f1
                    SGDFinalMacroF1 = [double]$SGD.final_metrics_avg_per_task.macro_f1
                    AdamSelectedLR = [double]$Adam.selected_lr
                    AdamAtLowerBoundary = ConvertTo-StrictBoolean $Adam.at_lower_boundary
                    AdamAtUpperBoundary = ConvertTo-StrictBoolean $Adam.at_upper_boundary
                    AdamFullyTunedClaim = ConvertTo-StrictBoolean $Adam.fully_tuned_claim_allowed
                }
            }
        }
    }

    $Rows |
        Sort-Object Variant, Attack, Seed |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/validation_scorecard.csv"

    $Summary = $Rows |
        Group-Object Variant |
        ForEach-Object {
            [PSCustomObject]@{
                Variant = $_.Name
                MeanMetaStep1 = ($_.Group.MetaStep1MacroF1 | Measure-Object -Average).Average
                MeanStep1GapVsSGD = ($_.Group.MetaMinusSGDStep1 | Measure-Object -Average).Average
                MeanMetaCurveAUC = ($_.Group.MetaCurveAUC | Measure-Object -Average).Average
                MeanCurveGapVsSGD = ($_.Group.MetaMinusSGDCurveAUC | Measure-Object -Average).Average
                MeanMetaFinalMacroF1 = ($_.Group.MetaFinalMacroF1 | Measure-Object -Average).Average
                AdamBoundaryHitCount = @(
                    $_.Group | Where-Object {
                        $_.AdamAtLowerBoundary -or $_.AdamAtUpperBoundary
                    }
                ).Count
            }
        } |
        Sort-Object Variant

    $Summary |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/validation_summary.csv"

    $BotnetRows = foreach ($Seed in @(42, 62)) {
        $ResultPath = "$Root/E0_final_only/botnet/seed_$Seed/horizon_20/validation/results.json"
        $Json = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
        $Exp = $Json.exp1_5shot
        foreach ($Split in @("validation", "test")) {
            $Info = $Exp.task_manifests.$Split
            $Stats = $Info.reuse_statistics
            [PSCustomObject]@{
                Attack = "botnet"
                Seed = $Seed
                Split = $Split
                SampledTasks = [int]$Stats.task_count
                UniqueTaskHashes = [int]$Stats.unique_task_hashes
                WindowOccurrences = [int]$Stats.window_occurrences
                UniqueWindows = [int]$Stats.unique_windows
                WindowReuseRate = [double]$Stats.window_reuse_rate
                RawDisjointTaskCountGreedy = [int]$Stats.raw_disjoint_task_count_greedy
                IndependentClaimAllowed = ConvertTo-StrictBoolean $Info.independent_replication_claim_allowed
            }
        }
    }

    $BotnetRows |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/botnet_task_independence.csv"

    Write-Host "`nValidation summary:" -ForegroundColor Green
    $Summary | Format-Table -AutoSize

    Write-Host "`nBotnet task independence:" -ForegroundColor Green
    $BotnetRows | Format-Table -AutoSize

    $BoundaryFailures = @(
        $Rows | Where-Object {
            $_.AdamAtLowerBoundary -or $_.AdamAtUpperBoundary
        }
    )
    if ($BoundaryFailures.Count -gt 0) {
        Write-Warning "Adam still hits an LR-grid boundary in $($BoundaryFailures.Count) validation runs."
        $BoundaryFailures |
            Select-Object Variant, Attack, Seed, AdamSelectedLR, AdamAtLowerBoundary, AdamAtUpperBoundary |
            Format-Table -AutoSize
    } else {
        Write-Host "PASS: Adam does not hit an LR-grid boundary." -ForegroundColor Green
    }
}

function Write-TestReports {
    $TestVariants = @("E0_final_only", $ChosenVariant) | Select-Object -Unique
    $TestRows = foreach ($Variant in $TestVariants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $ResultPath = "$Root/$Variant/$Attack/seed_$Seed/horizon_20/test_once/results.json"
                $Json = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
                $Exp = $Json.exp1_5shot
                $Meta = $Exp.methods.MetaOpt
                $SGD = $Exp.methods.SGD
                [PSCustomObject]@{
                    Variant = $Variant
                    Attack = $Attack
                    Seed = $Seed
                    MetaStep1 = [double]$Meta.adaptation_analysis.checkpoints.'1'.macro_f1.mean
                    SGDStep1 = [double]$SGD.adaptation_analysis.checkpoints.'1'.macro_f1.mean
                    MetaCurveAUC = [double]$Meta.adaptation_analysis.curve_auc_mean
                    SGDCurveAUC = [double]$SGD.adaptation_analysis.curve_auc_mean
                    MetaFinalMacroF1 = [double]$Meta.final_metrics_avg_per_task.macro_f1
                    SGDFinalMacroF1 = [double]$SGD.final_metrics_avg_per_task.macro_f1
                    MetaNonfiniteCount = [int]$Meta.nonfinite_count
                }
            }
        }
    }

    $GateRows = foreach ($Row in ($TestRows | Where-Object Variant -eq $ChosenVariant)) {
        $E0 = $TestRows |
            Where-Object {
                $_.Variant -eq "E0_final_only" -and
                $_.Attack -eq $Row.Attack -and
                $_.Seed -eq $Row.Seed
            } |
            Select-Object -First 1

        [PSCustomObject]@{
            Attack = $Row.Attack
            Seed = $Row.Seed
            Step1GainVsE0 = $Row.MetaStep1 - $E0.MetaStep1
            Step1GapVsSGD = $Row.MetaStep1 - $Row.SGDStep1
            CurveAUCGapVsSGD = $Row.MetaCurveAUC - $Row.SGDCurveAUC
            FinalDeltaVsE0 = $Row.MetaFinalMacroF1 - $E0.MetaFinalMacroF1
            NonfiniteCount = $Row.MetaNonfiniteCount
            PassStep1Gain = ($Row.MetaStep1 - $E0.MetaStep1) -ge 0.08
            PassStep1Gap = ($Row.MetaStep1 - $Row.SGDStep1) -ge -0.05
            PassCurveGap = ($Row.MetaCurveAUC - $Row.SGDCurveAUC) -ge -0.02
            PassFinalRegression = ($Row.MetaFinalMacroF1 - $E0.MetaFinalMacroF1) -ge -0.01
            PassFinite = $Row.MetaNonfiniteCount -eq 0
        }
    }

    $GateRows |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/test_success_gates.csv"

    $DynamicsRows = @()
    $DynamicsGates = @()
    foreach ($Variant in $TestVariants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $UpdatePath = "$Root/$Variant/$Attack/seed_$Seed/horizon_20/test_once/update_analysis.csv"
                $Rows = Import-Csv -LiteralPath $UpdatePath |
                    Where-Object { $_.group -eq "all" -and $_.method -eq "MetaOpt" }
                if (-not $Rows) {
                    throw "No aggregate MetaOpt update rows found: $UpdatePath"
                }

                $StepMedians = foreach ($Group in ($Rows | Group-Object step)) {
                    [PSCustomObject]@{
                        Variant = $Variant
                        Attack = $Attack
                        Seed = $Seed
                        Step = [int]$Group.Name
                        MedianGradientNorm = Get-Median @($Group.Group | ForEach-Object { [double]$_.grad_norm })
                        MedianUpdateNorm = Get-Median @($Group.Group | ForEach-Object { [double]$_.update_norm })
                        MedianUpdateGradRatio = Get-Median @($Group.Group | ForEach-Object { [double]$_.update_to_grad_ratio })
                        MeanCosineUpdateGrad = ($Group.Group | ForEach-Object { [double]$_.cosine_update_grad } | Measure-Object -Average).Average
                        ClipRatio = ($Group.Group | ForEach-Object { [double]$_.was_clipped } | Measure-Object -Average).Average
                        TrustLimitedRatio = ($Group.Group | ForEach-Object { [double]$_.was_trust_limited } | Measure-Object -Average).Average
                        MedianAnchorUpdateNorm = Get-Median @($Group.Group | ForEach-Object { [double]$_.anchor_update_norm })
                        MedianResidualUpdateNorm = Get-Median @($Group.Group | ForEach-Object { [double]$_.residual_update_norm })
                        MedianGate = Get-Median @($Group.Group | ForEach-Object { [double]$_.gate_mean })
                    }
                }
                $DynamicsRows += $StepMedians

                $PositiveRatios = @(
                    $StepMedians |
                        ForEach-Object { [double]$_.MedianUpdateGradRatio } |
                        Where-Object { $_ -gt 0 }
                )
                if ($PositiveRatios.Count -gt 0) {
                    $RatioMin = ($PositiveRatios | Measure-Object -Minimum).Minimum
                    $RatioMax = ($PositiveRatios | Measure-Object -Maximum).Maximum
                    $RatioSpan = $RatioMax / $RatioMin
                } else {
                    $RatioSpan = [double]::PositiveInfinity
                }

                $ClipRatio = ($Rows | ForEach-Object { [double]$_.was_clipped } | Measure-Object -Average).Average
                $Numeric = @(
                    $Rows | ForEach-Object {
                        [double]$_.grad_norm
                        [double]$_.update_norm
                        [double]$_.update_to_grad_ratio
                        [double]$_.raw_update_norm
                    }
                )
                $Nonfinite = @(
                    $Numeric | Where-Object {
                        [double]::IsNaN($_) -or [double]::IsInfinity($_)
                    }
                ).Count

                $DynamicsGates += [PSCustomObject]@{
                    Variant = $Variant
                    Attack = $Attack
                    Seed = $Seed
                    OverallClipRatio = $ClipRatio
                    MedianRatioSpan = $RatioSpan
                    NonfiniteDynamicsValues = $Nonfinite
                    PassClipRatio = $ClipRatio -lt 0.05
                    PassRatioWithinTwoOrders = $RatioSpan -le 100.0
                    PassFiniteDynamics = $Nonfinite -eq 0
                }
            }
        }
    }

    $DynamicsRows |
        Sort-Object Variant, Attack, Seed, Step |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/update_dynamics_by_step.csv"
    $DynamicsGates |
        Sort-Object Variant, Attack, Seed |
        Export-Csv -NoTypeInformation -Encoding UTF8 "$Root/update_dynamics_gates.csv"

    Write-Host "`nTest success gates:" -ForegroundColor Green
    $GateRows | Format-Table -AutoSize
    Write-Host "`nUpdate/clip dynamics gates:" -ForegroundColor Green
    $DynamicsGates | Format-Table -AutoSize
}

Assert-RepositoryRoot

if ($Stage -eq "Check") {
    Get-Command $PythonCommand -ErrorAction Stop | Out-Null
    Assert-Cuda
    Write-Host "Environment check passed." -ForegroundColor Green
    Write-Host "Next command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts/run_metaopt_minimal_experiments.ps1 -Stage Prepare"
    exit 0
}

if ($Stage -eq "Prepare") {
    Assert-Cuda
    if (-not (Test-Path -LiteralPath $Root)) {
        New-Item -ItemType Directory -Path $Root | Out-Null
    } elseif (-not $Resume) {
        throw "$Root already exists. Use -Resume only for the same experiment run, or choose another -Root."
    }

    foreach ($Variant in $Variants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $RunDir = "$Root/$($Variant.Id)/$Attack/seed_$Seed/horizon_20"
                $Artifact = "$RunDir/meta_artifacts.pt"
                if (Assert-CanCreateOrResume $Artifact "training artifact") {
                    Write-Host "`n=== TRAIN $($Variant.Id) / $Attack / seed=$Seed ===" -ForegroundColor Cyan
                    $TrainArgs = @(
                        "train_meta.py",
                        "--config", "configs/base.yaml",
                        "--dataset", "configs/datasets/cicids2017.yaml",
                        "--out", $Artifact,
                        "--override",
                        "device.prefer=cuda:0",
                        "experiment.seed=$Seed",
                        "data.unknown_class=$Attack",
                        "data.known_classes=$($KnownClasses[$Attack])",
                        "data.k_shot=5",
                        "meta.inner_steps=20",
                        "train.meta_epochs=10",
                        "train.early_stopping.enabled=true",
                        "train.early_stopping.patience=4",
                        "train.validation.checkpoints=[1,2,5,10,20]",
                        "adaptation_speed.max_steps=20",
                        "adaptation_speed.checkpoints=[0,1,2,5,10,20]"
                    )
                    $TrainArgs += $Variant.Extra
                    Invoke-Python -PyArgs $TrainArgs
                }

                Invoke-Python -PyArgs @(
                    "scripts/verify_metaopt_checkpoint.py",
                    "--artifacts", $Artifact,
                    "--best", "$RunDir/checkpoints/best.pt",
                    "--last", "$RunDir/checkpoints/last.pt",
                    "--out", "$RunDir/p0_checkpoint_scope_receipt.json"
                )
            }
        }
    }

    foreach ($Attack in @("ddos", "botnet")) {
        foreach ($Seed in @(42, 62)) {
            $Artifact = "$Root/E0_final_only/$Attack/seed_$Seed/horizon_20/meta_artifacts.pt"
            $ManifestDir = "$Root/manifests/$Attack/seed_$Seed"
            $ValidationManifest = "$ManifestDir/validation_5shot.json"
            $TestManifest = "$ManifestDir/test_5shot.json"

            if (Assert-CanCreateOrResume $ValidationManifest "validation manifest") {
                Invoke-Python -PyArgs @(
                    "scripts/generate_eval_task_manifest.py",
                    "--artifacts", $Artifact,
                    "--out", $ValidationManifest,
                    "--shot", "5",
                    "--tasks", "30",
                    "--task-seed", "$($Seed + 1)",
                    "--split", "val"
                )
            }
            if (Assert-CanCreateOrResume $TestManifest "test manifest") {
                Invoke-Python -PyArgs @(
                    "scripts/generate_eval_task_manifest.py",
                    "--artifacts", $Artifact,
                    "--out", $TestManifest,
                    "--shot", "5",
                    "--tasks", "100",
                    "--task-seed", "$($Seed + 1001)",
                    "--split", "test"
                )
            }
        }
    }

    foreach ($Variant in $AllowedVariants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $RunDir = "$Root/$Variant/$Attack/seed_$Seed/horizon_20"
                $ManifestDir = "$Root/manifests/$Attack/seed_$Seed"
                $ResultPath = "$RunDir/validation/results.json"
                if (Assert-CanCreateOrResume $ResultPath "validation result") {
                    Write-Host "`n=== VALIDATION $Variant / $Attack / seed=$Seed ===" -ForegroundColor Cyan
                    Invoke-Python -PyArgs @(
                        "scripts/run_experiments.py",
                        "--artifacts", "$RunDir/meta_artifacts.pt",
                        "--out", "$RunDir/validation",
                        "--phase", "validation",
                        "--validation-task-manifest", "$ManifestDir/validation_5shot.json",
                        "--test-task-manifest", "$ManifestDir/test_5shot.json",
                        "--override",
                        "device.prefer=cuda:0",
                        "compare.shots=[5]",
                        "compare.val_tasks=30",
                        "compare.test_tasks=100",
                        "adaptation_speed.max_steps=20",
                        "adaptation_speed.checkpoints=[0,1,2,5,10,20]",
                        "compare.baseline_lr_grid.sgd=[0.5,0.1,0.05,0.01]",
                        "compare.baseline_lr_grid.adam=[0.001,0.003,0.01,0.03,0.1,0.3]"
                    )
                }
            }
        }
    }

    Write-ValidationReports
    Write-Host "`nPrepare stage completed. Review:" -ForegroundColor Green
    Write-Host "  $Root/validation_summary.csv"
    Write-Host "  $Root/validation_scorecard.csv"
    Write-Host "  $Root/botnet_task_independence.csv"
    Write-Host "`nThen run exactly one frozen test stage:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts/run_metaopt_minimal_experiments.ps1 -Stage Test -ChosenVariant E1_multistep"
    exit 0
}

if ($Stage -eq "Test") {
    Assert-Cuda
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "Prepare output not found: $Root"
    }
    if ($ChosenVariant -notin $AllowedVariants) {
        throw "-ChosenVariant must be one of: $($AllowedVariants -join ', ')"
    }

    $ChoicePath = "$Root/frozen_test_choice.json"
    if (Test-Path -LiteralPath $ChoicePath) {
        $ExistingChoice = Get-Content -LiteralPath $ChoicePath -Raw | ConvertFrom-Json
        if ($ExistingChoice.chosen_variant -ne $ChosenVariant) {
            throw "Test choice is already frozen as $($ExistingChoice.chosen_variant); refusing to change it after test access."
        }
        if (-not $Resume) {
            throw "Frozen test choice already exists. Use -Resume only to continue the same choice."
        }
    } else {
        [PSCustomObject]@{
            chosen_variant = $ChosenVariant
            selection_source = "validation_only"
            test_used_for_selection = $false
            frozen_at = (Get-Date).ToString("o")
        } |
            ConvertTo-Json |
            Set-Content -LiteralPath $ChoicePath -Encoding UTF8
    }

    $TestVariants = @("E0_final_only", $ChosenVariant) | Select-Object -Unique
    foreach ($Variant in $TestVariants) {
        foreach ($Attack in @("ddos", "botnet")) {
            foreach ($Seed in @(42, 62)) {
                $RunDir = "$Root/$Variant/$Attack/seed_$Seed/horizon_20"
                $ManifestDir = "$Root/manifests/$Attack/seed_$Seed"
                $ResultPath = "$RunDir/test_once/results.json"
                if (Assert-CanCreateOrResume $ResultPath "one-time test result") {
                    Write-Host "`n=== ONE-TIME TEST $Variant / $Attack / seed=$Seed ===" -ForegroundColor Yellow
                    Invoke-Python -PyArgs @(
                        "scripts/run_experiments.py",
                        "--artifacts", "$RunDir/meta_artifacts.pt",
                        "--out", "$RunDir/test_once",
                        "--phase", "test",
                        "--selection-receipt", "$RunDir/validation/validation_selection.json",
                        "--validation-task-manifest", "$ManifestDir/validation_5shot.json",
                        "--test-task-manifest", "$ManifestDir/test_5shot.json",
                        "--override",
                        "device.prefer=cuda:0",
                        "compare.shots=[5]",
                        "compare.val_tasks=30",
                        "compare.test_tasks=100",
                        "adaptation_speed.max_steps=20",
                        "adaptation_speed.checkpoints=[0,1,2,5,10,20]"
                    )
                }
            }
        }
    }

    Write-TestReports
    Write-Host "`nTest stage completed. Final evidence:" -ForegroundColor Green
    Write-Host "  $Root/test_success_gates.csv"
    Write-Host "  $Root/update_dynamics_by_step.csv"
    Write-Host "  $Root/update_dynamics_gates.csv"
}
