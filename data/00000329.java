public class RequestTest { @Test public void createsADescriptionWithANameForClasses() { Description description = Request .classes(RequestTest.class, RequestTest.class).getRunner() .getDescription(); assertThat(description.toString(), is("classes")); } @Test public void reportsInitializationErrorThrownWhileCreatingSuite() { EventCollector collector = new EventCollector(); JUnitCore core = new JUnitCore(); core.addListener(collector); core.run(new FailingComputer(), FooTest.class, BarTest.class); assertThat(collector, hasSingleFailureWithMessage("cannot create suite")); } private static class FailingComputer extends Computer { @Override public Runner getSuite(RunnerBuilder builder, Class<?>[] classes) throws InitializationError { throw new InitializationError("cannot create suite"); } } private static class FooTest { } private static class BarTest { } }