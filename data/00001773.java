public class TapBalanceHashJoinSameSourceTransformer extends RuleInsertionTransformer { public TapBalanceHashJoinSameSourceTransformer() { super( BalanceAssembly, new BalanceHashJoinSameSourceExpression(), ElementCapture.Secondary, IntermediateTapElementFactory.TEMP_TAP ); } }