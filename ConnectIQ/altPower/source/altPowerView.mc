import Toybox.Activity;
import Toybox.Lang;
import Toybox.Time;
import Toybox.WatchUi;
import Toybox.Math;

class altPowerView extends WatchUi.SimpleDataField {

    // Set the label of the data field here.
    function initialize() {
        SimpleDataField.initialize();
        label = "AltPower";
        // units = "aW";
    }

    // The given info object contains all the current workout
    // information. Calculate a value and return it in this method.
    // Note that compute() and onUpdate() are asynchronous, and there is no
    // guarantee that compute() will be called before onUpdate().
    function compute(info as Activity.Info) as Numeric or Duration or String or Null {
        // See Activity.Info in the documentation for available information.
        var p = info.currentPower;
        if (p==null){p = 0;}
        var a = info.altitude/1000.0;
        var aP = p*(-1.12*a*a - 1.90*a + 99.9)/100.00;
        var pString = Math.round(aP).toString();
        var aString = (Math.round((-1.12*a*a - 1.90*a + 99.9))/100.0).toString();
        return (-1.12*a*a - 1.90*a + 99.9)/100.00;
    }

}
