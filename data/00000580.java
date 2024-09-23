public class SuiteMethod extends JUnit38ClassRunner { public SuiteMethod(Class<?> klass) throws Throwable { super(testFromSuiteMethod(klass)); } public static Test testFromSuiteMethod(Class<?> klass) throws Throwable { Method suiteMethod = null; Test suite = null; try { suiteMethod = klass.getMethod("suite"); if (!Modifier.isStatic(suiteMethod.getModifiers())) { throw new Exception(klass.getName() + ".suite() must be static"); } suite = (Test) suiteMethod.invoke(null); } catch (InvocationTargetException e) { throw e.getCause(); } return suite; } }