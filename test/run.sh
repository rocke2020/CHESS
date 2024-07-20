# 
file=test/execute_sql.py
python $file \
    2>&1  </dev/null | tee $file.log