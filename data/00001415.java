public class ApplyDebugLevelTransformer extends RuleContractedTransformer { public ApplyDebugLevelTransformer ( ) { super ( PreResolveAssembly , new PlannerLevelExpression ( DebugLevel . class ) ) ; } }