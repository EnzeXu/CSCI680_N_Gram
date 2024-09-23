public class ListenableFutureTest { private ExecutorService service = Executors.newCachedThreadPool(); @Test public void verifyOnComplete() throws Exception { DummyListenableFuture<String> future = new DummyListenableFuture<String>(false, service); final CountDownLatch latch = new CountDownLatch(1); future.addListener(new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { assertEquals("Hello World", (String) future.get()); latch.countDown(); } }); future.set("Hello World"); assertTrue(latch.await(1, TimeUnit.SECONDS)); } @Test public void verifyOnCompleteWhenAlreadyDone() throws Exception { DummyListenableFuture<String> future = new DummyListenableFuture<String>(true, service); final CountDownLatch latch = new CountDownLatch(1); future.addListener(new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { latch.countDown(); } }); assertTrue(latch.await(1, TimeUnit.SECONDS)); } @Test public void verifyOnCompleteWhenCancelled() throws Exception { DummyListenableFuture<String> future = new DummyListenableFuture<String>(false, service); final CountDownLatch latch = new CountDownLatch(1); future.addListener(new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { assertTrue(future.isCancelled()); latch.countDown(); } }); future.cancel(true); assertTrue(latch.await(1, TimeUnit.SECONDS)); } @Test public void verifyRemoval() throws Exception { DummyListenableFuture<String> future = new DummyListenableFuture<String>(false, service); final CountDownLatch latch = new CountDownLatch(1); final GenericCompletionListener listener = new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { latch.countDown(); } }; future.addListener(listener); future.removeListener(listener); Thread.sleep(500); assertEquals(1, latch.getCount()); } @Test public void verifyMultipleListeners() throws Exception { DummyListenableFuture<String> future = new DummyListenableFuture<String>(false, service); final CountDownLatch latch = new CountDownLatch(2); final GenericCompletionListener listener1 = new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { latch.countDown(); } }; final GenericCompletionListener listener2 = new GenericCompletionListener() { @Override public void onComplete(Future future) throws Exception { latch.countDown(); } }; future.addListener(listener1); future.addListener(listener2); future.set("Hello World"); assertTrue(latch.await(1, TimeUnit.SECONDS)); } }