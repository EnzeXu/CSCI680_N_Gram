public class BoundaryBalanceHashJoinSameSourceTransformer extends RuleInsertionTransformer { public BoundaryBalanceHashJoinSameSourceTransformer ( ) { super ( BalanceAssembly , new BalanceHashJoinSameSourceExpression ( ) , ElementCapture . Secondary , BoundaryElementFactory . BOUNDARY_PIPE ) ; } }