from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import NEO4J_CONFIG


class IndexUtil:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG['uri'],
            username=NEO4J_CONFIG['auth'][0],
            password=NEO4J_CONFIG['auth'][1]
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name='BAAI/bge-base-zh-v1.5',
            # 归一化向量
            encode_kwargs={"normalize_embeddings": True}
        )

    def create_fulltext_index(self, index_name, label, label_property):
        """
        :param index_name: 索引名称
        :param label: 节点
        :param label_property: 节点的属性名称
        """
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.{label_property}]
        """
        # 查询
        self.graph.query(cypher)

    def create_vector_index(self, label, source_property, index_name, embedding_property):
        """
        :param label: 节点标签
        :param source_property: 元属性：文本内容
        :param index_name: 索引名称
        :param embedding_property: 嵌入向量名称：属性值
        :return:
        """
        embedding_dim = self._add_embedding(label, source_property, embedding_property)
        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{embedding_property})
        OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {embedding_dim},
        `vector.similarity_function`: 'cosine'
        }}
        }}
        """
        self.graph.query(cypher)

    def _add_embedding(self, label, source_property, embedding_property):
        # 查询节点
        cypher = f"""
                    match (n:{label}) 
                    return (n.{source_property}) as text,elementId(n) as id
                  """
        # 取出节点的name值
        results = self.graph.query(cypher)
        texts = [result['text'] for result in results]
        ids = [result['id'] for result in results]
        # 调用模型生成embedding
        embeddings = self.embedding.embed_documents(texts)
        batch = []
        for id, embedding in zip(ids, embeddings):
            item = {"id": id, "embedding": embedding}
            batch.append(item)
        # 在节点增加新的属性:向量
        cypher = f"""
             UNWIND $batch as batch
             MATCH (n:{label}) 
             WHERE elementId(n) = batch.id
             SET n.{embedding_property} = batch.embedding
        """
        self.graph.query(cypher, params={"batch": batch})
        return len(embeddings[0])


if __name__ == '__main__':
    index = IndexUtil()
    index.create_fulltext_index("Trademark_fulltext_index", "Trademark", "name")
    index.create_vector_index(index_name="Trademark_vector_index",
                              label="Trademark",
                              source_property="name",
                              embedding_property="embedding")

    index_name = "Trademark_vector_index"  # default index name
    keyword_index_name = "Trademark_fulltext_index"  # default keyword index name

    store = Neo4jVector.from_existing_index(
        index.embedding,
        url=NEO4J_CONFIG['uri'],
        username=NEO4J_CONFIG['auth'][0],
        password=NEO4J_CONFIG['auth'][1],
        index_name=index_name,
        keyword_index_name=keyword_index_name,
        search_type=SearchType.HYBRID,
    )
    res = store.similarity_search("Apple", k=5)[0].page_content
    print(res)

    # 创建其他的索引
    # SPU
    index.create_fulltext_index("SPU_fulltext_index", "SPU", "name")
    index.create_vector_index(index_name="SPU_vector_index",
                              label="SPU",
                              source_property="name",
                              embedding_property="embedding")
    # SKU
    index.create_fulltext_index("SKU_fulltext_index", "SKU", "name")
    index.create_vector_index(index_name="SKU_vector_index",
                              label="SKU",
                              source_property="name",
                              embedding_property="embedding")
    # Category1
    index.create_fulltext_index("Category1_fulltext_index", "Category1", "name")
    index.create_vector_index(index_name="Category1_vector_index",
                              label="Category1",
                              source_property="name",
                              embedding_property="embedding")
    # Category2
    index.create_fulltext_index("Category2_fulltext_index", "Category2", "name")
    index.create_vector_index(index_name="Category2_vector_index",
                              label="Category2",
                              source_property="name",
                              embedding_property="embedding")
    # Category3
    index.create_fulltext_index("Category3_fulltext_index", "Category3", "name")
    index.create_vector_index(index_name="Category3_vector_index",
                              label="Category3",
                              source_property="name",
                              embedding_property="embedding")
