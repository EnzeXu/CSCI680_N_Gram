public class BlockJUnit4ClassRunnerWithParametersFactory implements ParametersRunnerFactory { public Runner createRunnerForTestWithParameters(TestWithParameters test) throws InitializationError { return new BlockJUnit4ClassRunnerWithParameters(test); } }