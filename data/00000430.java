public class ClassRequestTest { public static class PrivateSuiteMethod { static junit.framework.Test suite() { return null; } } @Test public void noSuiteMethodIfMethodPrivate() throws Throwable { assertNull(new SuiteMethodBuilder() .runnerForClass(PrivateSuiteMethod.class)); } }