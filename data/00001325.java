public class PlannerContext { public static final PlannerContext NULL = new PlannerContext ( ) ; RuleRegistry ruleRegistry ; FlowPlanner flowPlanner ; FlowDef flowDef ; Flow flow ; boolean isTransformTracingEnabled = false ; private Map properties ; public PlannerContext ( ) { this . properties = new Properties ( ) ; } public PlannerContext ( RuleRegistry ruleRegistry ) { this . ruleRegistry = ruleRegistry ; } public PlannerContext ( RuleRegistry ruleRegistry , FlowPlanner flowPlanner , FlowDef flowDef , Flow flow , boolean isTransformTracingEnabled ) { this . ruleRegistry = ruleRegistry ; this . flowPlanner = flowPlanner ; this . flowDef = flowDef ; this . flow = flow ; this . isTransformTracingEnabled = isTransformTracingEnabled ; if ( flowPlanner != null ) this . properties = flowPlanner . getDefaultProperties ( ) ; else this . properties = new Properties ( ) ; } public String getStringProperty ( String property ) { return PropertyUtil . getStringProperty ( System . getProperties ( ) , properties , property ) ; } public int getIntProperty ( String property , int defaultValue ) { return PropertyUtil . getIntProperty ( System . getProperties ( ) , properties , property , defaultValue ) ; } public RuleRegistry getRuleRegistry ( ) { return ruleRegistry ; } public FlowPlanner getFlowPlanner ( ) { return flowPlanner ; } public FlowDef getFlowDef ( ) { return flowDef ; } public Flow getFlow ( ) { return flow ; } public ProcessLogger getLogger ( ) { Flow flow = getFlow ( ) ; if ( flow == null ) return ProcessLogger . NULL ; return ( ProcessLogger ) flow ; } public boolean isTransformTracingEnabled ( ) { return isTransformTracingEnabled ; } public PlannerLevel getPlannerLevelFor ( Class < ? extends PlannerLevel > plannerLevelClass ) { Map < Class < ? extends PlannerLevel > , PlannerLevel > levels = new HashMap < > ( ) ; addLevel ( levels , flowPlanner . getDebugLevel ( flowDef ) ) ; addLevel ( levels , flowPlanner . getAssertionLevel ( flowDef ) ) ; return levels . get ( plannerLevelClass ) ; } private void addLevel ( Map < Class < ? extends PlannerLevel > , PlannerLevel > levels , PlannerLevel level ) { if ( level != null ) levels . put ( level . getClass ( ) , level ) ; } public ElementFactory getElementFactoryFor ( String factoryName ) { return ruleRegistry . getElementFactory ( factoryName ) ; } }