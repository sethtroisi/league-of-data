time="${1:-30}"
echo "older than $time minutes"

if compgen -G "/tmp/tmp-tf-lol/*"; then
    echo "deleting some file"
    find /tmp/tmp-tf-lol/* -type d -cmin "+$time" -prune -exec rm -rf {} \;
else
    echo "No files to delete"
fi
