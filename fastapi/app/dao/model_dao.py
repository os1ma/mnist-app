import MySQLdb


class ModelDao:
    def find_by_tag(self, tag: str):
        try:
            con = MySQLdb.connect(
                host='db',
                db='app',
                user='app',
                passwd='password',
                charset='utf8')

            cur= con.cursor()
            sql = 'select id, tag, created_at from models where tag = %s'
            cur.execute(sql, (tag,))
            rows = cur.fetchall()

            if len(rows) == 0:
                return None

            row = rows[0]
            model = {
              'id': row[0],
              'tag': row[1],
              'created_at': row[2]
            }
            return model

        finally:
            cur.close()
            con.close()

    def insert(self, tag: str) -> None:
        try:
            con = MySQLdb.connect(
                host='db',
                db='app',
                user='app',
                passwd='password',
                charset='utf8')

            cur= con.cursor()
            sql = 'insert into models (tag) values (%s)'
            cur.execute(sql, (tag,))
            con.commit()

        finally:
            cur.close()
            con.close()
