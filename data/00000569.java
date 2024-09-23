public class ClassRequest extends MemoizingRequest { private final Class < ? > fTestClass ; private final boolean canUseSuiteMethod ; public ClassRequest ( Class < ? > testClass , boolean canUseSuiteMethod ) { this . fTestClass = testClass ; this . canUseSuiteMethod = canUseSuiteMethod ; } public ClassRequest ( Class < ? > testClass ) { this ( testClass , true ) ; } @ Override protected Runner createRunner ( ) { return new CustomAllDefaultPossibilitiesBuilder ( ) . safeRunnerForClass ( fTestClass ) ; } private class CustomAllDefaultPossibilitiesBuilder extends AllDefaultPossibilitiesBuilder { @ Override protected RunnerBuilder suiteMethodBuilder ( ) { return new CustomSuiteMethodBuilder ( ) ; } } private class CustomSuiteMethodBuilder extends SuiteMethodBuilder { @ Override public Runner runnerForClass ( Class < ? > testClass ) throws Throwable { if ( testClass == fTestClass && !canUseSuiteMethod ) { return null ; } return super . runnerForClass ( testClass ) ; } } }