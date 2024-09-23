public class Hadoop3MRFlowConnector extends FlowConnector { public Hadoop3MRFlowConnector() { } @ConstructorProperties({"properties"}) public Hadoop3MRFlowConnector( Map<Object, Object> properties ) { super( properties ); } @ConstructorProperties({"ruleRegistrySet"}) public Hadoop3MRFlowConnector( RuleRegistrySet ruleRegistrySet ) { super( ruleRegistrySet ); } @ConstructorProperties({"properties", "ruleRegistrySet"}) public Hadoop3MRFlowConnector( Map<Object, Object> properties, RuleRegistrySet ruleRegistrySet ) { super( properties, ruleRegistrySet ); } @Override protected Class<? extends Scheme> getDefaultIntermediateSchemeClass() { return SequenceFile.class; } @Override protected FlowPlanner createFlowPlanner() { return new Hadoop3MRPlanner(); } @Override protected RuleRegistrySet createDefaultRuleRegistrySet() { return new RuleRegistrySet( new MapReduceHadoopRuleRegistry() ); } }