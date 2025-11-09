param(
    [Parameter(Mandatory=$true)]
    [string]$FunctionName
)



# This script runs a specified function based on a parameter.

# Define all your functions first.

function t00 {
    Write-Host "--- Running t00 (Base Test) ---"
    python test_np.py
    python test_pt.py
    python diff.py >> result.log
}

function t01 {
    Write-Host "--- Running t01 ---"
    python test01_np.py
    python test01_pt.py
    python diff.py >> result.log
}

function t03 {
    Write-Host "--- Running t03 ---"
    python test03_np.py
    python test03_pt.py
    python diff.py >> result.log
}

function t04 {
    Write-Host "--- Running t04 ---"
    python test04_np.py
    python test04_pt.py
    python diff.py >> result.log
}

function t05 {
    Write-Host "--- Running t05 ---"
    python test05_np.py
    python test05_pt.py
    python diff.py >> result.log
}

function ALL {
    Write-Host "--- Running ALL Tests ---"
    t00
    t01
    t03
    t04
    t05
    re
}

function re{
    Write-Host "--- Displaying result.log ---"
    python eval.py
}

function cl{
    Write-Host "--- Cleaning up result.log ---"
    Remove-Item result.log -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------
# Define the script parameters
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Logic to execute the specified function
# ------------------------------------------------------------------

if (Get-Command -Name $FunctionName -CommandType Function -ErrorAction SilentlyContinue) {
    Write-Host "Executing function: $($FunctionName)" -ForegroundColor Green
    
    # Use the Call Operator (&) to dynamically execute the function
    & $FunctionName
    
    Write-Host "Function $($FunctionName) finished." -ForegroundColor Green
}
else {
    Write-Host "Error: Function '$($FunctionName)' not found." -ForegroundColor Red
    Write-Host "Available functions are: t00, t01, t03, t04, t05" -ForegroundColor Yellow
    exit 1
}