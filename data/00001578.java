public class TapBalanceCheckpointTransformer extends RuleInsertionTransformer { public TapBalanceCheckpointTransformer ( ) { super ( BalanceAssembly , new BalanceCheckpointWithTapExpression ( ) , IntermediateTapElementFactory . TEMP_TAP ) ; } }