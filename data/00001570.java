public class RuleAssert extends GraphAssert<ElementGraph> implements Rule { private final PlanPhase phase; private final RuleExpression ruleExpression; private ContractedTransformer contractedTransformer; private SubGraphTransformer subGraphTransformer; public RuleAssert( PlanPhase phase, RuleExpression ruleExpression, String message ) { this( phase, ruleExpression, message, null ); } public RuleAssert( PlanPhase phase, RuleExpression ruleExpression, String message, AssertionType assertionType ) { super( ruleExpression.getMatchExpression(), message, assertionType ); this.phase = phase; this.ruleExpression = ruleExpression; if( ruleExpression.getContractionExpression() != null ) contractedTransformer = new ContractedTransformer( ruleExpression.getContractionExpression() ); else contractedTransformer = null; if( ruleExpression.getContractedMatchExpression() != null ) { if( contractedTransformer == null ) throw new IllegalArgumentException( "must have contracted expression if given contracted match expression" ); subGraphTransformer = new SubGraphTransformer( contractedTransformer, ruleExpression.getContractedMatchExpression() ); } else { subGraphTransformer = null; } } @Override public PlanPhase getRulePhase() { return phase; } @Override public String getRuleName() { return getClass().getSimpleName().replaceAll( "^(.*)[A-Z][a-z]*Rule$", "$1" ); } @Override protected Transformed<ElementGraph> transform( PlannerContext plannerContext, ElementGraph graph ) { Transformed transformed = null; if( contractedTransformer != null ) transformed = contractedTransformer.transform( plannerContext, graph ); else if( subGraphTransformer != null ) transformed = subGraphTransformer.transform( plannerContext, graph ); return transformed; } @Override public String toString() { return getRuleName(); } }