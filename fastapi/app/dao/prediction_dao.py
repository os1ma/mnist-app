import json

from app.dao.dao_utils import MySQLConnection


class PredictionDao:
    def find_all(self):
        with MySQLConnection() as db:
            sql = 'select id, model_id, image_id, result, created_at from predictions order by id'
            db.cur.execute(sql)
            rows = db.cur.fetchall()

            predictions = []
            for row in rows:
                prediction = {
                    'id': row[0],
                    'model_id': row[1],
                    'image_id': row[2],
                    'result': row[3],
                    'created_at': row[4]
                }
                predictions.append(prediction)

            return predictions

    def insert(self, model_id, image_id, result) -> None:
        with MySQLConnection() as db:
            sql = 'insert into predictions (model_id, image_id, result) values (%s, %s, %s)'
            db.cur.execute(sql, (model_id, image_id, json.dumps(result)))
            db.con.commit()
            return db.cur.lastrowid
