from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import *

load_dotenv(ROOT_PATH / '.env')


class ChatService:
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
        # 初始化大模型
        self.model = init_chat_model(
            model='deepseek-ai/DeepSeek-V3.2',
            model_provider='openai',
            api_key=os.getenv('MODEL_SCOPE_API_KEY'),
            base_url=os.getenv("MODEL_SCOPE_BASE_URL"),
            temperature=0
        )
        # 创建查询store
        self.store = {
            'Trademark': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='Trademark_vector_index',
                keyword_index_name='Trademark_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'SPU': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='SPU_vector_index',
                keyword_index_name='SPU_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'SKU': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='SKU_vector_index',
                keyword_index_name='SKU_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category1': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='Category1_vector_index',
                keyword_index_name='Category1_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category2': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='Category2_vector_index',
                keyword_index_name='Category2_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category3': Neo4jVector.from_existing_index(
                self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name='Category3_vector_index',
                keyword_index_name='Category3_fulltext_index',
                search_type=SearchType.HYBRID,
            )
        }
        # 大模型输入
        self.json_parser = JsonOutputParser()
        # 输出大模型
        self.str_parser = StrOutputParser()

    def _generate_cypher(self, question):
        """
        根据用户的问题生成查询Cypher,在Cypher的Label部分使用变量占位
        :param question:用户的问题
        :return: 模型输出
        """
        template = """
                你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

                用户问题：{question}

                知识图谱结构信息：{schema_info}

                要求：
                1. 生成参数化Cypher查询语句，用param_0, param_1等代替具体值,
                2. 识别需要对齐的实体
                3. 必须严格使用以下JSON格式输出结果
                {{
                  "cypher_query": "生成的Cypher语句",
                  "entities_to_align": [
                    {{
                      "param_name": "param_0",
                      "entity": "原始实体名称",
                      "label": "节点类型"
                    }}
                  ]
                }}
        """
        prompt = (PromptTemplate.from_template(template=template)
                  .format_prompt(question=question, schema_info=self.graph.schema))
        output = self.model.invoke(prompt)
        res = self.json_parser.invoke(output)
        return res

    # 实体对齐
    def _entity_align(self, entities_to_align):
        """
        实体对齐,大模型输出的数据中的实体信息对齐到真实的信息
        :param entities_to_align: 需要对齐的实体
        :return: 对齐之后的实体
        """
        # 批处理
        for index, entity_to_align in enumerate(entities_to_align):
            label = entity_to_align['label']
            entity = entity_to_align['entity']
            aligned_entity = self.store[label].similarity_search(entity, k=1)[0].page_content
            # 只修改entity,实现对齐
            entities_to_align[index]['entity'] = aligned_entity

        return entities_to_align

    def _execute_cypher(self, cypher_query, aligned_entities):
        """
        将查询语句传入,格式化执查询语句,实体是对齐后的实体
        :param cypher_query: 查询语句
        :param aligned_entities: 对齐之后的实体
        :return:得到查询输出
        """
        params = {aligned_entity['param_name']:
                      aligned_entity['entity'] for aligned_entity in aligned_entities}
        return self.graph.query(cypher_query, params=params)

    def _generate_final_answer(self, question, query_result):
        """
        根据查询的输出,调用大模型得到最终的回答
        :param question: 用户的问题
        :param query_result: 查询输出
        :return:最终回答
        """
        template = """
                你是一个电商智能客服，根据用户问题，以及数据库查询结果生成一段简洁、准确的自然语言回答。
                用户问题: {question}
                数据库返回结果: {query_result}
                """
        prompt = PromptTemplate.from_template(template).format_prompt(question=question,
                                                                      query_result=query_result)
        # 调用大模型生成回答
        output = self.model.invoke(prompt)
        return self.str_parser.invoke(output)

    def chat(self, question):
        # 根据用户的问题,生成Cypher和需要对齐的实体
        result = self._generate_cypher(question)
        # 实体对齐(混合索引)
        entities_to_align = result['entities_to_align']
        cypher = result['cypher_query']
        aligned_entities = self._entity_align(entities_to_align)
        # 执行Cypher得到结果
        res = self._execute_cypher(cypher, aligned_entities)
        # 根据用户的输入和执行结果返回答案
        final_answer = self._generate_final_answer(question, res)
        return final_answer


if __name__ == '__main__':
    res = ChatService().chat("Apple 有哪些商品?")
