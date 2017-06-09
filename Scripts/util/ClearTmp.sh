if compgen -G "/tmp/tmp*"; then
    echo "deleting some file"
    find /tmp/tmp* -type d -cmin '+30' -exec rm -rf {} \;
else
    echo "No files to delete"
fi
