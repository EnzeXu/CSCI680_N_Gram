public class JUnit4Builder extends RunnerBuilder { @Override public Runner runnerForClass(Class<?> testClass) throws Throwable { return new JUnit4(testClass); } }