public class ToStringTest extends TestCase { public void testDefaultConnectionFactory() { (new DefaultConnectionFactory()).toString(); (new DefaultConnectionFactory(10, 1000)).toString(); (new DefaultConnectionFactory(100, 100, DefaultHashAlgorithm.KETAMA_HASH)).toString(); } public void testBinaryConnectionFactory() { (new BinaryConnectionFactory()).toString(); (new BinaryConnectionFactory(10, 1000)).toString(); (new BinaryConnectionFactory(100, 1000, DefaultHashAlgorithm.KETAMA_HASH)).toString(); } }