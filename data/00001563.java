public class IgnoresAnnotationsElementGraph extends DecoratedElementGraph { public IgnoresAnnotationsElementGraph( ElementGraph decorated ) { super( decorated ); } @Override public boolean equals( Object obj ) { return ElementGraphs.equalsIgnoreAnnotations( this, (ElementGraph) obj ); } @Override public int hashCode() { return ElementGraphs.hashCodeIgnoreAnnotations( this ); } }