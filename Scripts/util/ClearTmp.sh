if compgen -G "/tmp/tmp-tf-lol/*"; then
    echo "deleting some file"
    find /tmp/tmp-tf-lol/* -type d -cmin '+30' -exec rm -rf {} \;
else
    echo "No files to delete"
fi
