public class RuleChainTest { private static final List<String> LOG = new ArrayList<String>(); private static class LoggingRule extends TestWatcher { private final String label; public LoggingRule(String label) { this.label = label; } @Override protected void starting(Description description) { LOG.add("starting " + label); } @Override protected void finished(Description description) { LOG.add("finished " + label); } } public static class UseRuleChain { @Rule public final RuleChain chain = outerRule(new LoggingRule("outer rule")) .around(new LoggingRule("middle rule")).around( new LoggingRule("inner rule")); @Test public void example() { assertTrue(true); } } @Test public void executeRulesInCorrectOrder() throws Exception { testResult(UseRuleChain.class); List<String> expectedLog = asList("starting outer rule", "starting middle rule", "starting inner rule", "finished inner rule", "finished middle rule", "finished outer rule"); assertEquals(expectedLog, LOG); } @Test public void aroundShouldNotAllowNullRules() { RuleChain chain = RuleChain.emptyRuleChain(); try { chain.around(null); fail("around() should not allow null rules"); } catch (NullPointerException e) { assertThat(e.getMessage(), equalTo("The enclosed rule must not be null")); } } public static class RuleChainWithNullRules { @Rule public final RuleChain chain = outerRule(new LoggingRule("outer rule")) .around(null); @Test public void example() {} } @Test public void whenRuleChainHasNullRuleTheStacktraceShouldPointToIt() { Result result = JUnitCore.runClasses(RuleChainWithNullRules.class); assertThat(result.getFailures().size(), equalTo(1)); String stacktrace = Throwables.getStacktrace(result.getFailures().get(0).getException()); assertThat(stacktrace, containsString("\tat org.junit.rules.RuleChainTest$RuleChainWithNullRules.<init>(RuleChainTest.java:")); } }