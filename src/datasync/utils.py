import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

from configuration.config import *


class MysqlReader:
    def __init__(self):
        self.connection = pymysql.Connection(**MySQL_CONFIG)
        self.cursor = self.connection.cursor(DictCursor)

    def read(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.connection.close()


class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    def write_nodes(self, label: str, nodes_data):
        cypher = f"""
                    UNWIND $data AS item
                    MERGE (n:{label} {{id:item.id, name:item.name}})
                 """
        self.driver.execute_query(cypher, data=nodes_data)

    def write_relations(self, relation_label: str, start_node: str, end_node: str, nodes_data):
        cypher = f"""
                 UNWIND $data AS item
                 MATCH (start:{start_node} {{ id:item.start_id }}) , (end:{end_node} {{id:item.end_id }})
                 MERGE (start)-[:{relation_label}]->(end)          
                """
        self.driver.execute_query(cypher, data=nodes_data)

    def close(self):
        self.driver.close()


if __name__ == '__main__':
    reader = MysqlReader()
    writer = Neo4jWriter()
    # 写入一级分类
    sql = """
          select *
          from gmall.base_category1
          """
    node_data = reader.read(sql)
    writer.write_nodes("Category1", node_data)
    # 写入二级分类
    sql = """
          select *
          from gmall.base_category2
          """
    node_data = reader.read(sql)
    writer.write_nodes("Category2", node_data)
    # 写入关系
    sql = """
          select id as start_id, category1_id as end_id
          from gmall.base_category2
          """
    node_data = reader.read(sql)
    writer.write_relations("Belong", "Category2", "Category1", node_data)
