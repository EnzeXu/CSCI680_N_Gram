public class RMesh extends RGeomElem { public int type = RGeomElem . MESH ; public RStrip [ ] strips ; int currentStrip=0 ; public RMesh ( ) { strips = null ; type = RGeomElem . MESH ; } public RMesh ( RMesh m ) { if ( m == null ) { return ; } for ( int i=0 ; i < m . countStrips ( ) ; i++ ) { this . append ( new RStrip ( m . strips [ i ] ) ) ; } type = RGeomElem . MESH ; setStyle ( m ) ; } public int countStrips ( ) { if ( this . strips==null ) { return 0 ; } return this . strips . length ; } public void addStrip ( RStrip s ) { this . append ( s ) ; } public void addStrip ( ) { this . append ( new RStrip ( ) ) ; } public void setCurrent ( int indStrip ) { this . currentStrip = indStrip ; } public void addPoint ( RPoint p ) { if ( strips == null ) { this . append ( new RStrip ( ) ) ; } this . strips [ currentStrip ] . append ( p ) ; } public void addPoint ( float x , float y ) { if ( strips == null ) { this . append ( new RStrip ( ) ) ; } this . strips [ currentStrip ] . append ( new RPoint ( x , y ) ) ; } public void addPoint ( int indStrip , RPoint p ) { if ( strips == null ) { this . append ( new RStrip ( ) ) ; } this . strips [ indStrip ] . append ( p ) ; } public void addPoint ( int indStrip , float x , float y ) { if ( strips == null ) { this . append ( new RStrip ( ) ) ; } this . strips [ indStrip ] . append ( new RPoint ( x , y ) ) ; } public void draw ( PGraphics g ) { for ( int i=0 ; i < this . countStrips ( ) ; i++ ) { g . beginShape ( PConstants . TRIANGLE_STRIP ) ; if ( this . style . texture != null ) { g . texture ( this . style . texture ) ; for ( int j=0 ; j < this . strips [ i ] . vertices . length ; j++ ) { float x = this . strips [ i ] . vertices [ j ] . x ; float y = this . strips [ i ] . vertices [ j ] . y ; g . vertex ( x , y , x , y ) ; } } else { for ( int j=0 ; j < this . strips [ i ] . vertices . length ; j++ ) { float x = this . strips [ i ] . vertices [ j ] . x ; float y = this . strips [ i ] . vertices [ j ] . y ; g . vertex ( x , y ) ; } } g . endShape ( PConstants . CLOSE ) ; } } public void draw ( PApplet g ) { for ( int i=0 ; i < this . countStrips ( ) ; i++ ) { g . beginShape ( PConstants . TRIANGLE_STRIP ) ; if ( this . style . texture != null ) { g . texture ( this . style . texture ) ; } for ( int j=0 ; j < this . strips [ i ] . vertices . length ; j++ ) { g . vertex ( this . strips [ i ] . vertices [ j ] . x , this . strips [ i ] . vertices [ j ] . y ) ; } g . endShape ( PConstants . CLOSE ) ; } } public RPoint [ ] getHandles ( ) { int numStrips = countStrips ( ) ; if ( numStrips == 0 ) { return null ; } RPoint [ ] result=null ; RPoint [ ] newresult=null ; for ( int i=0 ; i < numStrips ; i++ ) { RPoint [ ] newPoints = strips [ i ] . getHandles ( ) ; if ( newPoints!=null ) { if ( result==null ) { result = new RPoint [ newPoints . length ] ; System . arraycopy ( newPoints , 0 , result , 0 , newPoints . length ) ; } else { newresult = new RPoint [ result . length + newPoints . length ] ; System . arraycopy ( result , 0 , newresult , 0 , result . length ) ; System . arraycopy ( newPoints , 0 , newresult , result . length , newPoints . length ) ; result = newresult ; } } } return result ; } public RPoint [ ] getPoints ( ) { int numStrips = countStrips ( ) ; if ( numStrips == 0 ) { return null ; } RPoint [ ] result=null ; RPoint [ ] newresult=null ; for ( int i=0 ; i < numStrips ; i++ ) { RPoint [ ] newPoints = strips [ i ] . getPoints ( ) ; if ( newPoints!=null ) { if ( result==null ) { result = new RPoint [ newPoints . length ] ; System . arraycopy ( newPoints , 0 , result , 0 , newPoints . length ) ; } else { newresult = new RPoint [ result . length + newPoints . length ] ; System . arraycopy ( result , 0 , newresult , 0 , result . length ) ; System . arraycopy ( newPoints , 0 , newresult , result . length , newPoints . length ) ; result = newresult ; } } } return result ; } public RPoint getPoint ( float t ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public RPoint getTangent ( float t ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public RPoint [ ] getTangents ( ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public RPoint [ ] [ ] getPointsInPaths ( ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public RPoint [ ] [ ] getHandlesInPaths ( ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public RPoint [ ] [ ] getTangentsInPaths ( ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return null ; } public boolean contains ( RPoint p ) { PApplet . println ( "Feature not yet implemented for this class . " ) ; return false ; } public int getType ( ) { return type ; } public void transform ( RMatrix m ) { int numStrips = countStrips ( ) ; if ( numStrips!=0 ) { for ( int i=0 ; i < numStrips ; i++ ) { strips [ i ] . transform ( m ) ; } } } public RMesh toMesh ( ) { return this ; } public RPolygon toPolygon ( ) throws RuntimeException { throw new RuntimeException ( "Transforming a Mesh to a Polygon is not yet implemented . " ) ; } public RShape toShape ( ) throws RuntimeException { throw new RuntimeException ( "Transforming a Mesh to a Shape is not yet implemented . " ) ; } void clear ( ) { this . strips = null ; } void append ( RStrip nextstrip ) { RStrip [ ] newstrips ; if ( strips==null ) { newstrips = new RStrip [ 1 ] ; newstrips [ 0 ] = nextstrip ; currentStrip = 0 ; } else { newstrips = new RStrip [ this . strips . length+1 ] ; System . arraycopy ( this . strips , 0 , newstrips , 0 , this . strips . length ) ; newstrips [ this . strips . length ] =nextstrip ; currentStrip++ ; } this . strips=newstrips ; } }