import MySQLdb


class MySQLConnection:
    def __enter__(self):
        self.con = MySQLdb.connect(
            host='db',
            db='app',
            user='app',
            passwd='password',
            charset='utf8')

        self.cur = self.con.cursor()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.cur.close()
        self.con.close()

    def commit(self):
        self.con.commit()
