# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import psycopg2
import psycopg2.extras

class DataBase():
    """
    database(Postgresql)にクエリを送って情報を取得するインターフェイス
    SQLが使えない人向け
    代表的な情報の取得のみ記述
    細かいクエリは各自が記述するように
    """
    def __init__(self, dbname, host, user, password, table_name="object_info"):
        self._table_name = table_name
        self.conn = psycopg2.connect(
            "dbname={0} host={1} user={2} password={3}".format(dbname, host, user, password))

    def get_objects_info(self):
        """
        DataBaseに保存されているobjectの情報をすべて取得する
        """
        with self.conn:
            sql_data = pd.read_sql(
                """
                SELECT
                  *
                FROM {};
                """.format(self._table_name),
                self.conn)
        return sql_data

    def delete_data(self):
        """
        tableのデータをすべて消去する
        """
        with self.conn:
            with self.conn.cursor() as curs:
                curs.execute(
                    """
                    DELETE FROM {};
                    """.format(self._table_name),
                    self.conn)

    def get_objects_info_filtered_id(self, specific_id):
        """
        情報を取得する
        idでfilter
        """
        with self.conn:
            sql_data = pd.read_sql(
                """
                SELECT
                  *
                FROM {0}
                WHERE specific_id is {1};
                """.format(self._table_name, specific_id),
                self.conn)
        return sql_data

    def get_objects_info_filtered_time(self, tstart, tend):
        """
        情報を取得する
        timeでfilter
        tstart : rospy.time 始まり
        tend : rospy.time 終わり
        """
        with self.conn:
            sql_data = pd.read_sql(
                """
                SELECT
                  *
                FROM {0}
                WHERE
                  {1} <= ros_timestamp AND
                  ros_timestamp <= {2}
                """.format(self._table_name, tstart, tend),
                self.conn)
        return sql_data

    def get_objects_info_latest(self):
        """
        databaseに入っている最新の情報のみ取り出す
        """
        with self.conn:
            sql_data = pd.read_sql(
                """
                SELECT
                  *
                FROM {}
                WHERE
                  ros_timestamp > 0 AND
                  ros_timestamp = (SELECT max(ros_timestamp) FROM object_info)
                """.format(self._table_name),
                self.conn)
        return sql_data


if __name__ == '__main__':
    db_name = "********"
    host = "192.168.***.****"
    user = "********"
    password = "*******"
    db = DataBase(db_name, host, user, password)
    print db.get_objects_info()
    # print db.get_objects_info_filtered_id(3)
