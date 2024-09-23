public class AssumingInTheoriesTest { @Test public void noTheoryAnnotationMeansAssumeShouldIgnore() { Assume.assumeTrue(false); } @Test public void theoryMeansOnlyAssumeShouldFail() throws InitializationError { Result result = runTheoryClass(TheoryWithNoUnassumedParameters.class); Assert.assertEquals(1, result.getFailureCount()); } public static class TheoryWithNoUnassumedParameters { @DataPoint public static final boolean FALSE = false; @Theory public void theoryWithNoUnassumedParameters(boolean value) { Assume.assumeTrue(value); } } }