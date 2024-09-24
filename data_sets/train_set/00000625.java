public class HostBeanTest { private static final String[] FIELDS = { "nickname", "username", "hostname", "port" }; private HostBean host1; private HostBean host2; @Before public void setUp() { host1 = new HostBean(); host1.setNickname("Home"); host1.setUsername("bob"); host1.setHostname("server.example.com"); host1.setPort(22); host2 = new HostBean(); host2.setNickname("Home"); host2.setUsername("bob"); host2.setHostname("server.example.com"); host2.setPort(22); } @Test public void id_Equality() { host1.setId(1); host2.setId(1); assertTrue(host1.equals(host2)); assertTrue(host1.hashCode() == host2.hashCode()); } @Test public void id_Inequality() { host1.setId(1); host2.setId(2); assertFalse("HostBeans are equal when their ID is different", host1 .equals(host2)); assertFalse("HostBean hash codes are equal when their ID is different", host1.hashCode() == host2.hashCode()); } @Test public void id_Equality2() { host1.setId(1); host2.setId(1); host2.setNickname("Work"); host2.setUsername("alice"); host2.setHostname("client.example.com"); assertTrue( "HostBeans are not equal when their ID is the same but other fields are different!", host1.equals(host2)); assertTrue( "HostBeans hashCodes are not equal when their ID is the same but other fields are different!", host1.hashCode() == host2.hashCode()); } @Test public void equals_Empty_Success() { HostBean bean1 = new HostBean(); HostBean bean2 = new HostBean(); assertEquals(bean1, bean2); } @Test public void equals_NicknameDifferent_Failure() { host1.setNickname("Work"); assertNotEquals(host1, host2); } @Test public void equals_NicknameNull_Failure() { host1.setNickname(null); assertNotEquals(host1, host2); } @Test public void equals_ProtocolNull_Failure() { host1.setProtocol(null); assertNotEquals(host1, host2); } @Test public void equals_ProtocolDifferent_Failure() { host1.setProtocol("fake"); assertNotEquals(host1, host2); } @Test public void equals_UserDifferent_Failure() { host1.setUsername("joe"); assertNotEquals(host1, host2); } @Test public void equals_UserNull_Failure() { host1.setUsername(null); assertNotEquals(host1, host2); } @Test public void equals_HostDifferent_Failure() { host1.setHostname("work.example.com"); assertNotEquals(host1, host2); } @Test public void equals_HostNull_Failure() { host1.setHostname(null); assertNotEquals(host1, host2); } @Test public void testBeanMeetsEqualsContract() { assertMeetsEqualsContract(HostBean.class, FIELDS); } @Test public void testBeanMeetsHashCodeContract() { assertMeetsHashCodeContract(HostBean.class, FIELDS); } }