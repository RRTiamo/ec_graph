from utils import MysqlReader, Neo4jWriter


class TableSync:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    def sync_category1(self):
        # 写入一级分类
        sql = """
              select id, name
              from gmall.base_category1
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("Category1", node_data)

    def sync_category2(self):
        # 写入二级分类
        sql = """
              select id, name
              from gmall.base_category2 \
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("Category2", node_data)

    def sync_category2_relation_category1(self):
        sql = """
              select id as start_id, category1_id as end_id
              from gmall.base_category2 \
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Belong", "Category2", "Category1", node_data)

    def sync_category3(self):
        # 写入三级级分类
        sql = """
              select id, name
              from gmall.base_category3
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("Category3", node_data)

    def sync_category3_relation_category2(self):
        sql = """
              select id as start_id, category2_id as end_id
              from gmall.base_category3
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Belong", "Category3", "Category2", node_data)

    def sync_base_attr_name(self):
        # 写入平台属性
        sql = """
              select id, attr_name as name
              from gmall.base_attr_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("AttrName", node_data)

    def sync_base_attr_value(self):
        # 写入平台属性
        sql = """
              select id, value_name as name
              from gmall.base_attr_value
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("AttrValue", node_data)

    def sync_attr_name_relation_attr_value(self):
        sql = """
              select id as end_id, attr_id as start_id
              from gmall.base_attr_value
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Have", "AttrName", "AttrValue", node_data)

    def sync_category1_relation_attr_name(self):
        sql = """
              select category_id as start_id, id as end_id
              from gmall.base_attr_info
              where category_level = 1
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Have", "Category1", "AttrName", node_data)

    def sync_category2_relation_attr_name(self):
        sql = """
              select category_id as start_id, id as end_id
              from gmall.base_attr_info
              where category_level = 2
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Have", "Category2", "AttrName", node_data)

    def sync_category3_relation_attr_name(self):
        sql = """
              select category_id as start_id, id as end_id
              from gmall.base_attr_info
              where category_level = 3
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Have", "Category3", "AttrName", node_data)

    def sync_spu(self):
        # 写入SPU节点
        sql = """
              select id, spu_name as name
              from gmall.spu_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("SPU", node_data)

    def sync_sku(self):
        # 写入SKU节点
        sql = """
              select id, sku_name as name
              from gmall.sku_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("SKU", node_data)

    def sync_sku_belong_spu(self):
        sql = """
              select id as start_id, spu_id as end_id
              from gmall.sku_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Belong", "SKU", "SPU", node_data)

    def sync_spu_belong_category3(self):
        sql = """
              select id as start_id, category3_id as end_id
              from gmall.spu_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Belong", "SPU", "Category3", node_data)

    def sync_trademark(self):
        # 写入trademark节点
        sql = """
              select id, tm_name as name
              from gmall.base_trademark
              """
        node_data = self.reader.read(sql)
        self.writer.write_nodes("Trademark", node_data)

    def sync_spu_belong_trademark(self):
        sql = """
              select id as start_id, tm_id as end_id
              from gmall.spu_info
              """
        node_data = self.reader.read(sql)
        self.writer.write_relations("Belong", "SPU", "Trademark", node_data)

    # TODO:写入销售属性
    def sync_sale_attr(self):
        sql = """
              select id,
                     sale_attr_name name
              from spu_sale_attr
              """
        self.writer.write_nodes('SaleAttr', self.reader.read(sql))

    def sync_sale_attr_spu(self):
        sql = """
              select id     end_id,
                     spu_id start_id
              from spu_sale_attr
              """
        relationships = self.reader.read(sql)
        self.writer.write_relations("Have", "SPU", "SaleAttr", relationships)

    def sync_sale_attr_value(self):
        sql = """
              select id,
                     sale_attr_value_name name
              from spu_sale_attr_value
              """
        self.writer.write_nodes("SaleAttrValue", self.reader.read(sql))

    def sync_sale_attr_value_attr(self):
        sql = """
              select a.id start_id,
                     v.id end_id
              from spu_sale_attr_value v
                       join spu_sale_attr a on v.spu_id = a.spu_id and v.base_sale_attr_id = a.base_sale_attr_id
              """
        relationships = self.reader.read(sql)
        self.writer.write_relations('Have', "SaleAttr", 'SaleAttrValue', relationships)


if __name__ == '__main__':
    table_sync = TableSync()

    # 写入分类属性
    table_sync.sync_category1()
    table_sync.sync_category2()
    table_sync.sync_category3()
    table_sync.sync_category2_relation_category1()
    table_sync.sync_category3_relation_category2()

    # 写入平台属性
    table_sync.sync_base_attr_name()
    table_sync.sync_base_attr_value()
    table_sync.sync_attr_name_relation_attr_value()
    table_sync.sync_category1_relation_attr_name()
    table_sync.sync_category2_relation_attr_name()
    table_sync.sync_category3_relation_attr_name()

    # 写入商品信息
    table_sync.sync_spu()
    table_sync.sync_sku()
    table_sync.sync_sku_belong_spu()
    table_sync.sync_spu_belong_category3()

    # 写入品牌信息
    table_sync.sync_trademark()
    table_sync.sync_spu_belong_trademark()

    # TODO:写入销售属性
    # table_sync.sync_sale_attr()
    # table_sync.sync_sale_attr_spu()
    # table_sync.sync_sale_attr_value()
    # table_sync.sync_sale_attr_value_attr()