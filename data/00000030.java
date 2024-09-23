public class WokenUpOnIdleTest { @ Test public void shouldWakeUpOnIdle ( ) throws Exception { CountDownLatch latch = new CountDownLatch ( 3 ) ; MemcachedConnection connection = new InstrumentedConnection ( latch , 1024 , new BinaryConnectionFactory ( ) , Arrays . asList ( new InetSocketAddress ( 11211 ) ) , Collections . < ConnectionObserver > emptyList ( ) , FailureMode . Redistribute , new BinaryOperationFactory ( ) ) ; assertTrue ( latch . await ( 5 , TimeUnit . SECONDS ) ) ; } static class InstrumentedConnection extends MemcachedConnection { final CountDownLatch latch ; InstrumentedConnection ( CountDownLatch latch , int bufSize , ConnectionFactory f , List < InetSocketAddress > a , Collection < ConnectionObserver > obs , FailureMode fm , OperationFactory opfactory ) throws IOException { super ( bufSize , f , a , obs , fm , opfactory ) ; this . latch = latch ; } @ Override protected void handleWokenUpSelector ( ) { latch . countDown ( ) ; } } }