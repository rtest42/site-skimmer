import subprocess

result = subprocess.run(["node", "test.js"], capture_output=True, text=True)
print(result.stdout)