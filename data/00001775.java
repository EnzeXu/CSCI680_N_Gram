public class TapBalanceGroupSplitTriangleTransformer extends RuleInsertionTransformer { public TapBalanceGroupSplitTriangleTransformer() { super( BalanceAssembly, new BalanceGroupSplitTriangleExpression(), IntermediateTapElementFactory.TEMP_TAP ); } }