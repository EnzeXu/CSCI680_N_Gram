public class TapBalanceGroupBlockingHashJoinTransformer extends RuleInsertionTransformer { public TapBalanceGroupBlockingHashJoinTransformer() { super( BalanceAssembly, new BalanceGroupBlockingHashJoinExpression(), IntermediateTapElementFactory.TEMP_TAP ); } }