public class LocalPlanner extends FlowPlanner<LocalFlow, Properties> { public static final String PLATFORM_NAME = "local"; public LocalPlanner() { } @Override public Properties getDefaultConfig() { return null; } @Override public PlannerInfo getPlannerInfo( String registryName ) { return new PlannerInfo( getClass().getSimpleName(), PLATFORM_NAME, registryName ); } @Override public PlatformInfo getPlatformInfo() { return new PlatformInfo( "local", "Chris K Wensel <chris@wensel.net>", Version.getRelease() ); } protected LocalFlow createFlow( FlowDef flowDef ) { return new LocalFlow( getPlatformInfo(), getDefaultProperties(), getDefaultConfig(), flowDef ); } @Override public FlowStepFactory<Properties> getFlowStepFactory() { return new BaseFlowStepFactory<Properties>( getFlowNodeFactory() ) { @Override public FlowStep<Properties> createFlowStep( ElementGraph stepElementGraph, FlowNodeGraph flowNodeGraph ) { return new LocalFlowStep( stepElementGraph, flowNodeGraph ); } }; } @Override protected Tap makeTempTap( String prefix, String name ) { return null; } }