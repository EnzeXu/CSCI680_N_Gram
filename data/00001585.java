public class TapBalanceNonSafeSplitTransformer extends RuleInsertionTransformer { public TapBalanceNonSafeSplitTransformer ( ) { super ( BalanceAssembly , new BalanceNonSafeSplitExpression ( ) , IntermediateTapElementFactory . TEMP_TAP ) ; } }