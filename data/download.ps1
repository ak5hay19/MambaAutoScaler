# Download Alibaba Cluster Trace v2018 machine_usage dataset
# Run from the AutoScaler project root

$url = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz"
$tarball = "data\machine_usage.tar.gz"
$dest = "data\machine_usage.csv"

if (Test-Path $dest) {
    Write-Host "Dataset already exists at $dest. Skipping download."
    exit 0
}

Write-Host "Downloading machine_usage.tar.gz (~1.7 GB)..."
Invoke-WebRequest -Uri $url -OutFile $tarball -UseBasicParsing

Write-Host "Extracting..."
# Use tar (available on Windows 10/11)
tar -xzf $tarball -C data\

if (Test-Path $dest) {
    Write-Host "Done. Dataset saved to $dest"
    Remove-Item $tarball -Force
} else {
    Write-Host "Extraction complete. Locate the .csv inside data\ and rename to machine_usage.csv"
    Get-ChildItem data\ -Filter "*.csv"
}
