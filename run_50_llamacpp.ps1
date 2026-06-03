$ErrorActionPreference = "Continue"
$base = $PSScriptRoot
$testDir = "$base\test-random50"
$outDir = "$base\output"
$cliBin = if ($env:LLAMA_BIN) { Join-Path $env:LLAMA_BIN "llama-mtmd-cli.exe" } else { "$env:USERPROFILE\.llama\bin\llama-mtmd-cli.exe" }
$prompt = "Extract ALL text from this image verbatim."

$models = [ordered]@{
    "lightonocr-2-1b" = "ggml-org/LightOnOCR-2-1B-GGUF:Q8_0"
    "olmocr-2-7b"     = "lmstudio-community/olmOCR-2-7B-1025-GGUF:Q4_K_M"
}

foreach ($modelName in $models.Keys) {
    $modelDir = "$outDir\$modelName\random50"
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    
    for ($i = 1; $i -le 50; $i++) {
        $img = "$testDir\test$i.png"
        $outFile = "$modelDir\test$i.md"
        
        Write-Host "[$modelName] test$i..."
        $stderrFile = "$env:TEMP\$modelName-test$i-err.txt"
        $result = & $cliBin -hf $models[$modelName] --image $img -p $prompt -n 512 --temp 0.1 -ngl 99 2>$stderrFile
        $result | Out-File -FilePath $outFile -Encoding utf8
        Write-Host "  => $($result.Length) chars"
    }
    Write-Host "[$modelName] done!"
}

Write-Host "ALL DONE!"
