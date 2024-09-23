public class AsciiIPV6ClientTest extends AsciiClientTest { @ Override protected void initClient ( ConnectionFactory cf ) throws Exception { client = new MemcachedClient ( cf , AddrUtil . getAddresses ( TestConfig . IPV6_ADDR + " : " + TestConfig . PORT_NUMBER ) ) ; } @ Override protected String getExpectedVersionSource ( ) { if ( TestConfig . defaultToIPV4 ( ) ) { return String . valueOf ( new InetSocketAddress ( TestConfig . IPV4_ADDR , TestConfig . PORT_NUMBER ) ) ; } return String . valueOf ( new InetSocketAddress ( TestConfig . IPV6_ADDR , TestConfig . PORT_NUMBER ) ) ; } }