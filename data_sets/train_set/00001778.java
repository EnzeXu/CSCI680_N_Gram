public class TapBalanceGroupGroupTransformer extends RuleInsertionTransformer { public TapBalanceGroupGroupTransformer() { super( BalanceAssembly, new BalanceGroupGroupExpression(), IntermediateTapElementFactory.TEMP_TAP ); } }