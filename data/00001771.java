public class TapBalanceHashJoinBlockingHashJoinTransformer extends RuleInsertionTransformer { public TapBalanceHashJoinBlockingHashJoinTransformer() { super( BalanceAssembly, new BalanceHashJoinBlockingHashJoinExpression(), IntermediateTapElementFactory.TEMP_TAP ); } }