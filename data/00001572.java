public class TapBalanceGroupSplitMergeGroupTransformer extends RuleInsertionTransformer { public TapBalanceGroupSplitMergeGroupTransformer ( ) { super ( BalanceAssembly , new BalanceGroupSplitMergeGroupExpression ( ) , IntermediateTapElementFactory . TEMP_TAP ) ; } }