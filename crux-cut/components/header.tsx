import Link from "next/link"
import { Button } from "@/components/ui/button"

export function Header() {
  return (
    <header className="border-b border-gray-800 bg-black/50 backdrop-blur-md sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
            CruxCut
          </span>
        </Link>
        <nav className="hidden md:flex items-center gap-6">
          <Link href="#features" className="text-sm text-gray-300 hover:text-white transition-colors">
            기능
          </Link>
          <Link href="#testimonials" className="text-sm text-gray-300 hover:text-white transition-colors">
            후기
          </Link>
          <Link href="#pricing" className="text-sm text-gray-300 hover:text-white transition-colors">
            가격
          </Link>
          <Link href="#faq" className="text-sm text-gray-300 hover:text-white transition-colors">
            FAQ
          </Link>
        </nav>
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            className="hidden md:flex border-gray-700 text-gray-300 hover:text-white hover:border-gray-500"
          >
            로그인
          </Button>
          <Button className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700">
            무료 체험
          </Button>
        </div>
      </div>
    </header>
  )
}
