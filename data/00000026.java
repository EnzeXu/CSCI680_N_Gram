public class KetamaConnectionFactoryTest extends TestCase { public void testCorrectTypes ( ) { ConnectionFactory factory = new KetamaConnectionFactory ( ) ; NodeLocator locator = factory . createLocator ( new ArrayList < MemcachedNode > ( ) ) ; assertTrue ( locator instanceof KetamaNodeLocator ) ; } public void testDefaultProperties ( ) { KetamaConnectionFactory connectionFactory = new KetamaConnectionFactory ( ) ; assertEquals ( connectionFactory . getHashAlg ( ) , DefaultHashAlgorithm . KETAMA_HASH ) ; assertTrue ( connectionFactory . getWeights ( ) . isEmpty ( ) ) ; assertEquals ( connectionFactory . getKetamaNodeKeyFormat ( ) , KetamaNodeKeyFormatter . Format . SPYMEMCACHED ) ; } public void testSettingProperties ( ) { Map < InetSocketAddress , Integer > weights = new HashMap < InetSocketAddress , Integer > ( ) ; weights . put ( new InetSocketAddress ( "localhost" , 11211 ) , 8 ) ; KetamaConnectionFactory connectionFactory = new KetamaConnectionFactory ( 1 , 1 , 1 , DefaultHashAlgorithm . FNV1_32_HASH , KetamaNodeKeyFormatter . Format . LIBMEMCACHED , weights ) ; assertEquals ( connectionFactory . getWeights ( ) , weights ) ; assertEquals ( connectionFactory . getHashAlg ( ) , DefaultHashAlgorithm . FNV1_32_HASH ) ; assertEquals ( connectionFactory . getKetamaNodeKeyFormat ( ) , KetamaNodeKeyFormatter . Format . LIBMEMCACHED ) ; } }