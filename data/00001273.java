public class FactoryLoader extends ProviderLoader < CascadingFactory > { private static FactoryLoader factoryLoader ; public synchronized static FactoryLoader getInstance ( ) { if ( factoryLoader == null ) factoryLoader = new FactoryLoader ( ) ; return factoryLoader ; } public < Factory extends CascadingFactory > Factory loadFactoryFrom ( FlowProcess flowProcess , String property , Class < Factory > defaultFactory ) { Object value = flowProcess . getProperty ( property ) ; String className ; if ( value == null ) className = defaultFactory . getName ( ) ; else className = value . toString ( ) ; Factory factory = ( Factory ) createProvider ( className ) ; if ( factory != null ) factory . initialize ( flowProcess ) ; return factory ; } }