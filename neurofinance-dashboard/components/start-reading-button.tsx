"use client";

import { useState } from "react";
import { Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface StartReadingButtonProps {
  onClick: () => void; // onClick is a function
  disabled?: boolean; // Optional disabled prop
}

export function StartReadingButton({ onClick, disabled }: StartReadingButtonProps) {
  const [isReading, setIsReading] = useState(false);

  const handleClick = async () => {
    setIsReading(true); // Set loading state
    try {
      await onClick(); // Call the onClick function passed as a prop
    } catch (error) {
      console.error("Error during reading:", error);
    } finally {
      setIsReading(false); // Reset loading state
    }
  };

  return (
    <Button
      size="lg"
      className="gap-2 px-8 py-2 font-bold w-full"
      onClick={handleClick}
      disabled={disabled || isReading} // Disable if loading or explicitly disabled
    >
      {isReading ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          READING...
        </>
      ) : (
        <>
          <Play className="h-4 w-4" />
          START READING
        </>
      )}
    </Button>
  );
}