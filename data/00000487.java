public class CustomBlockJUnit4ClassRunnerTest { @Test public void exceptionsFromMethodBlockMustNotResultInUnrootedTests() throws Exception { TrackingRunListener listener = new TrackingRunListener(); RunNotifier notifier = new RunNotifier(); notifier.addListener(listener); new CustomBlockJUnit4ClassRunner(CustomBlockJUnit4ClassRunnerTestCase.class).run(notifier); assertEquals("tests started.", 2, listener.testStartedCount.get()); assertEquals("tests failed.", 1, listener.testFailureCount.get()); assertEquals("tests finished.", 2, listener.testFinishedCount.get()); } public static class CustomBlockJUnit4ClassRunnerTestCase { @Test public void shouldPass() { } @Test public void throwException() { } } private static class CustomBlockJUnit4ClassRunner extends BlockJUnit4ClassRunner { CustomBlockJUnit4ClassRunner(Class<?> testClass) throws InitializationError { super(testClass); } @Override protected Statement methodBlock(FrameworkMethod method) { if ("throwException".equals(method.getName())) { throw new RuntimeException("throwException() test method invoked"); } return super.methodBlock(method); } } private static class TrackingRunListener extends RunListener { final AtomicInteger testStartedCount = new AtomicInteger(); final AtomicInteger testFailureCount = new AtomicInteger(); final AtomicInteger testFinishedCount = new AtomicInteger(); @Override public void testStarted(Description description) throws Exception { testStartedCount.incrementAndGet(); } @Override public void testFailure(Failure failure) throws Exception { testFailureCount.incrementAndGet(); } @Override public void testFinished(Description description) throws Exception { testFinishedCount.incrementAndGet(); } } }