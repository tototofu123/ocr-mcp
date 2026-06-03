$ErrorActionPreference = "Continue"
$base = $PSScriptRoot
$sources = "$base\sources"
$output = "$base\output"
$cliBin = if ($env:LLAMA_BIN) { Join-Path $env:LLAMA_BIN "llama-mtmd-cli.exe" } else { "$env:USERPROFILE\.llama\bin\llama-mtmd-cli.exe" }
$prompt = "Extract ALL text from this image verbatim. Preserve line breaks and structure. Output only the raw text content."

$models = [ordered]@{
    "lightonocr-2-1b"    = "ggml-org/LightOnOCR-2-1B-GGUF:Q8_0"
    "chandra-ocr-2"      = "prithivMLmods/chandra-ocr-2-GGUF:Q4_K_M"
    "dots-mocr"          = "lodrick-the-lafted/dots.mocr-gguf:q5_k_m"
    "chandra-v1"         = "prithivMLmods/chandra-OCR-GGUF:Q4_K_M"
    "olmocr-2-7b"        = "lmstudio-community/olmOCR-2-7B-1025-GGUF:Q4_K_M"
    "infinity-parser-7b" = "mradermacher/Infinity-Parser-7B-GGUF:Q4_K_M"
}

$total = 0; $failed = 0
foreach ($modelName in $models.Keys) {
    $modelDir = "$output\$modelName"
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    
    for ($i = 1; $i -le 10; $i++) {
        $img = "$sources\image$i.png"
        $outFile = "$modelDir\image$i-output.md"
        
        Write-Host "[$modelName] image$i..."
        $stderrFile = "$modelDir\image$i-stderr.txt"
        $result = & $cliBin -hf $models[$modelName] --image $img -p $prompt -n 512 --temp 0.1 -ngl 99 2>$stderrFile
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED (exit $LASTEXITCODE)"
            $failed++
        }
        $result | Out-File -FilePath $outFile -Encoding utf8
        $total++
        Write-Host "  => $(([string]$result).Length) chars"
    }
}
Write-Host "`n=== DONE: $total runs, $failed failed ==="
