time="${1:-30}"
echo "older than $time minutes"

if compgen -G "/tmp/tmp-tf-lol/exploring/*"; then
    echo "deleting some file"
    find /tmp/tmp-tf-lol/exploring/* -type d -cmin "+$time" -exec rm -rf {} \;
else
    echo "No files to delete"
fi
